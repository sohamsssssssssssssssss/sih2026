"""Single-process T09 runtime: shared lazy tools, per-tool locks and deadlines."""
import asyncio
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import resource
import subprocess
import threading
import time
from uuid import uuid4

from backend.ask_scenes import SceneStore
from backend.config import AskConfig, ROOT
from core.answer import assemble, decide, record_events
from core.gate import GateDecision, OUTSIDE_INDIA, Thresholds
from core import registry as registry_module
from core.schema import Answer, Evidence, Query, ToolResult
from orchestrator import trace

logger=logging.getLogger('uvicorn.error')


def memory():
    # macOS ru_maxrss is bytes; Linux reports KiB.
    import sys
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'peak_rss_bytes':int(peak if sys.platform=='darwin' else peak*1024)}


class RequestState:
    def __init__(self,timeout):
        self.trace_id=uuid4().hex
        self.deadline=time.monotonic()+timeout
        self.lock=threading.Lock()
        self.closed=False

    def expired(self):
        return self.closed or time.monotonic()>=self.deadline


class AskService:
    def __init__(self,config=None,registry=None):
        self.config=config or AskConfig()
        thresholds_payload=self.config.thresholds_file.read_bytes()
        self.thresholds=Thresholds.model_validate_json(thresholds_payload)
        self.thresholds_sha=hashlib.sha256(thresholds_payload).hexdigest()
        self.scenes=SceneStore(self.config)
        self.config.trace_dir.mkdir(parents=True,exist_ok=True)
        trace.TRACE_PATH=self.config.trace_dir/'trace.jsonl'
        self.registry=registry
        if self.registry is None:
            registry_module.register_default_tools()
            self.registry=registry_module._DEFAULT
        for tool in self.registry.list():
            if tool.name=='landcover': tool.cloud_threshold=self.thresholds.cloud_pct
        self.locks={t.name:threading.Lock() for t in self.registry.list()}
        self.loaded={}
        self.upload_lock=asyncio.Lock()
        self.executor=ThreadPoolExecutor(max_workers=4,thread_name_prefix='ask')
        try:
            self.commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError): self.commit='unknown'

    def close(self):
        self.executor.shutdown(wait=False,cancel_futures=True)

    def verified_trace(self,trace_id=None):
        with trace._LOCK:
            chain=trace.records()
            verified,message=trace.verify_chain()  # re-reads disk, not only the cached chain
            events=[r for r in chain if trace_id is None or r.get('trace_id')==trace_id]
        return {'events':events,'verified':verified,'message':message,'chain_scope':'entire persisted chain'}

    def outside_region(self):
        checked = self.verified_trace()
        if not checked['verified']:
            raise trace.TraceIntegrityError(checked['message'])
        trace_id = uuid4().hex
        summary = {'rule': 'region', 'outcome': 'out_of_region', 'policy': 'out_of_region',
                   'threshold': list(self.thresholds.india_bounds), 'detail': OUTSIDE_INDIA}
        record_events(trace_id, 'gate', [summary])
        gate = GateDecision(None, 'UNKNOWN', 'not_answerable', (Evidence(kind='text', summary=summary),))
        return assemble(None, gate, None, trace_id=trace_id)

    def work(self,request,state):
        first=self.scenes.load(request.scene_ref)
        query=Query(text=request.question,qtype='change' if request.second_scene_ref is not None else 'open')
        with state.lock:
            if state.expired(): return None
            checked=self.verified_trace()
            if not checked['verified']: raise trace.TraceIntegrityError(checked['message'])
            record_events(state.trace_id,'config',[{'thresholds_sha256':self.thresholds_sha,
                          'thresholds':self.thresholds.model_dump(mode='json'),'timeout_s':self.config.timeout}])
            record_events(state.trace_id,'scene',[{'scene_id':first.scene.scene_id,'source':first.scene.source,
                                                  **(first.provenance or {})}])
            _,gate=decide(query,first.scene,self.registry,trace_id=state.trace_id,
                          band_means=(first.means(),),
                          thresholds=self.thresholds)
            if not gate.passed:
                answer=assemble(None,gate,None,trace_id=state.trace_id)
                state.closed=True
                return answer
        tool=self.registry.get(gate.tool)
        lock=self.locks[tool.name]
        if not lock.acquire(timeout=max(0,state.deadline-time.monotonic())): return None
        try:
            if state.expired(): return None
            before=set(getattr(tool,'_loaded',{}))
            try:
                if tool.name=='landcover' and hasattr(tool,'_load'):
                    # Metadata gate passed; load under the tool lock before checking
                    # distribution shift against the checkpoint's own TRAIN statistics.
                    _,stats,_,_=tool._load(gate.mode.lower())
                    train={};means={}
                    for key,stat in stats.items():
                        values=first.arrays[key]['pixels'].astype(np.float32)
                        values=(np.clip(values,np.array(stat['clip_lo'])[:,None,None],np.array(stat['clip_hi'])[:,None,None])
                                if key=='s1' else values*stat['scale'])
                        for band,mean,std,observed in zip(first.arrays[key]['bands'],stat['mean'],stat['std'],values):
                            train[band]=(mean,std)
                            if np.isfinite(observed).any(): means[band]=float(observed[np.isfinite(observed)].mean())
                    after_load=set(getattr(tool,'_loaded',{}))
                    if after_load-before:
                        self.loaded[tool.name]=sorted(after_load)
                        logger.info('tool_load tool=%s modes=%s memory=%s',tool.name,self.loaded[tool.name],memory())
                    with state.lock:
                        if state.expired(): return None
                        if after_load-before:
                            record_events(state.trace_id,'load',[{'tool':tool.name,'modes':sorted(after_load-before),**memory()}])
                        _,gate=decide(query,first.scene,self.registry,trace_id=state.trace_id,
                                      band_means=(means,),
                                      training_stats=train,thresholds=self.thresholds)
                        if not gate.passed:
                            state.closed=True
                            return assemble(None,gate,None,trace_id=state.trace_id)
                    before=after_load
                pixels=first.pixels(tool.name,gate.mode,self.config.data_dir/'previews')
                with state.lock:
                    if state.expired(): return None
                    record_events(state.trace_id,'execution',[{'tool':tool.name,'mode':gate.mode,'outcome':'start'}])
                result=tool.run(first.scene,query,pixels)
            except Exception as error:
                result=ToolResult(tool=tool.name,tool_version=tool.version,latency_ms=0,
                                  value={'error':'Tool execution failed','error_type':type(error).__name__})
            after=set(getattr(tool,'_loaded',{}))
            if after-before or (tool.name not in self.loaded and isinstance(result,ToolResult)
                                 and not (isinstance(result.value,dict) and 'error' in result.value)):
                self.loaded[tool.name]=sorted(after) or ['ready']
                logger.info('tool_load tool=%s modes=%s memory=%s',tool.name,self.loaded[tool.name],memory())
            with state.lock:
                if state.expired():
                    if isinstance(result,ToolResult) and isinstance(result.value,dict) and 'error' in result.value:
                        record_events(state.trace_id,'execution',[{'outcome':'error_after_timeout','tool':tool.name,
                                      'error_type':result.value.get('error_type','ToolError')}])
                    return None  # timed-out native work retains the tool lock and cannot publish an answer
                if after-before:
                    record_events(state.trace_id,'load',[{'tool':tool.name,'modes':sorted(after-before),**memory()}])
                answer=assemble(result,gate,tool.status,trace_id=state.trace_id)
                state.closed=True
                return answer
        finally:
            lock.release()

    async def ask(self,request):
        state=RequestState(self.config.timeout)
        loop=asyncio.get_running_loop()
        future=loop.run_in_executor(self.executor,self.work,request,state)
        done,_=await asyncio.wait({future},timeout=self.config.timeout)
        if done:
            answer=future.result()
            if answer is not None: return answer
        # Python cannot safely kill a running native inference thread. Retain its
        # lock until completion, suppress late answers, and keep the HTTP deadline.
        if not done:
            future.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
        with state.lock:
            state.closed=True
            record_events(state.trace_id,'execution',[{'outcome':'timeout','timeout_s':self.config.timeout}])
            return assemble(None,GateDecision(None,'UNKNOWN','timeout',()),None,trace_id=state.trace_id)
