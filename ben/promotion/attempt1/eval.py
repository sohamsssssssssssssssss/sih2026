from pathlib import Path
import json, hashlib, warnings, sys
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from backend.ask_scenes import SceneStore
from backend.config import AskConfig,ROOT
from core.tools.landcover import LandcoverTool, CLASSES
warnings.simplefilter('error',ConvergenceWarning)
out=Path(__file__).parent
home=Path.home()/'ben-work'
run=home/'final-run-1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
archive=next((home/'adapt-run-3/seed-1').glob('*.npz'))
a=np.load(archive)
report=json.loads((run/'ben-adapt.json').read_text())
assert sha(archive)==json.loads((archive.parent/'ben-adapt.json').read_text())['test_predictions']['sha256']
assert a['seed']==1 and list(a['classes'])==list(CLASSES) and str(a['wc_manifest_sha256'])==report['wc_manifest_sha256']
cal={'source':str(archive.relative_to(home)),'fit_npz_sha256':sha(archive),'seed':1,'arm':'full','fit_fold':'A','clip':[1e-6,1-1e-6],'method':'Platt on logit(probability)','modes':{}}
def ece(p,y):
 bins=np.minimum((p*10).astype(int),9)
 return sum(abs((p[bins==i]-y[bins==i]).sum()) for i in range(10))/len(y)
lines=['# India land-cover promotion','', 'recipe validated on held-out regions; shipped weights have no held-out score','', 'Gate registration `4b4ee12`. Candidate code `4b1d57c`.','', '| Gate | Evidence | Decision |','|---|---|---|']
details=[]
passes=[]
def row(name,evidence,ok):
 passes.append(ok);lines.append(f'| {name} | {evidence} | {"PASS" if ok else "FAIL"} |')
for m,floor in [('s1',0.689087241566672),('s2',0.6588597201349471)]:
 ap=report['modes'][m]['out_of_fold']['full']['macro_map']
 row('G1 '+m,f'Recipe OOF mAP {ap}, floor {floor}',ap>=floor)
 mask=(a['mode']==m)&(a['arm']=='full')
 A=mask&(a['fold']=='A');B=mask&(a['fold']=='B')
 assert A.sum()==B.sum()==900 and len(set(a['patch_id'][A|B]))==1800
 params=cal['modes'][m]={};metrics=[]
 for j,c in enumerate(CLASSES):
  p=a['scores'][:,j].astype(float);y=a['labels'][:,j]
  x=logit(p.clip(1e-6,1-1e-6))[:,None]
  fit=LogisticRegression(C=1,solver='lbfgs',fit_intercept=True,max_iter=1000).fit(x[A],y[A])
  params[c]={'slope':float(fit.coef_[0,0]),'intercept':float(fit.intercept_[0])}
  raw,adj=ece(p[B],y[B]),ece(fit.predict_proba(x[B])[:,1],y[B])
  metrics.append((raw,adj));details.append(f'| {m} | {c} | {raw:.8f} | {adj:.8f} |')
 raw,adj=np.mean(metrics,axis=0);row('G2 '+m,f'Fold B macro ECE {raw:.8f} → {adj:.8f}',adj<raw)
dump(out/'calibrator.json',cal)
dump(out/'cfg.json',{'checkpoints':{'s2':str(run/'ben-landcover-india-s2.pt')}})
tool=LandcoverTool(out/'cfg.json')
for name,path,target in [('Ludhiana',ROOT/'data/fresh-backups/ludhiana-s2.tif','built'),('Punjab',ROOT/'tests/fixtures/punjab-s2.tif','cropland')]:
 s=SceneStore(AskConfig()).raster(path.read_bytes(),fresh_mode='cached' if name=='Ludhiana' else None)
 result=tool.run(s.scene,None,s.pixels('landcover','S2',out))
 scores=result.value['scores'];params=cal['modes']['s2']
 pred={c:expit(logit(np.clip(scores[c],1e-6,1-1e-6))*params[c]['slope']+params[c]['intercept']) for c in CLASSES}
 ranked=sorted(pred,key=pred.get,reverse=True)
 row('G3 '+name,', '.join(f'{c} {pred[c]:.8f}' for c in ranked[:3]),target in ranked[:2])
lines+=['','**Joint decision: '+('PASS' if all(passes) else 'FAIL. Retain the Europe checkpoints.')+'**','','G3 uses two one-scene sanity checks. G2 fits each class on 900 seed-1 fold A OOF rows and evaluates on 900 distinct fold B rows. These calibrators transfer from the fold models to the final weights.','','| Mode | Class | Raw ECE | Calibrated ECE |','|---|---|---:|---:|',*details,'','## Verified SHA-256','',f'- Fit NPZ: `{sha(archive)}`',*[f'- {p.name}: `{sha(p)}`' for p in sorted(run.glob('*.pt'))]]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:17]))
