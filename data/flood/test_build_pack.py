"""Tiny synthetic flood-pack tests. No network and no external data required."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data.flood import build_pack as b


@pytest.fixture
def raw(tmp_path):
    root = tmp_path/'raw'
    (root/'splits').mkdir(parents=True)
    (root/'sources').mkdir()
    for kind in ('source','label'):
        (root/'sources'/f'hand-{kind}-collection.json').write_text('{"license":"proprietary"}')
    split_ids = {'train':['India_1','India_2'],'valid':['Ghana_3'],'test':['India_4'],'bolivia':['Bolivia_5']}
    for split,ids in split_ids.items():
        (root/'splits'/b.SPLITS[split]).write_text(''.join(f'{c}_S1Hand.tif,{c}_LabelHand.tif\n' for c in ids))
        for chip in ids:
            for layer in b.LAYERS:
                folder=root/'layers'/layer; folder.mkdir(parents=True,exist_ok=True)
                bands=len(b.BANDS.get(layer,[None]))
                value=(2 if chip.endswith('_1') else 4) if split=='train' else 1000
                if layer=='S2Hand':
                    value=100 if split=='train' else 10000
                if layer not in b.BANDS:
                    value=0
                array=np.full((bands,8,8),value,dtype=b.SOURCE_DTYPES[layer])
                if layer in ('LabelHand','S1OtsuLabelHand'):
                    array[:,0,0]=-1; array[:,1:3,:]=1
                if layer=='JRCWaterHand':
                    array[:,1:3,:]=1
                nodata={'S1Hand':float('nan'),'S2Hand':0,'S1OtsuLabelHand':-1}.get(layer)
                if layer=='S1Hand':
                    array[:,0,0]=np.nan
                if layer=='S2Hand':
                    array[:,0,0]=0
                    if split!='train':
                        array[0,0,1]=32767
                with rasterio.open(folder/f'{chip}_{layer}.tif','w',driver='GTiff',height=8,width=8,
                                   count=bands,dtype=array.dtype,crs='EPSG:4326',
                                   transform=from_origin(-1,10,.0001,.0001),nodata=nodata) as ds:
                    ds.write(array)
                    if layer in b.BANDS:
                        ds.descriptions=tuple(b.BANDS[layer])
    log={str(p.relative_to(root)):{'url':'https://synthetic.invalid/'+str(p.relative_to(root)),
                                  'sha256':b.sha256(p),'bytes':p.stat().st_size}
         for p in root.rglob('*') if p.is_file()}
    (root/'FETCH_LOG.json').write_text(json.dumps(log))
    return root


@pytest.fixture
def pack(raw,tmp_path):
    output=tmp_path/'pack'
    manifest=b.assemble(raw,output,'a'*40,chip_size=8)
    return output,manifest


def test_decode_shapes_dtypes_and_encoding(raw):
    for layer in b.LAYERS:
        array,valid,meta=b.decode(raw/'layers'/layer/f'India_1_{layer}.tif',layer,8)
        assert array.shape==(len(b.BANDS.get(layer,[None])),8,8)
        expected='int8' if layer in ('LabelHand','S1OtsuLabelHand') else b.SOURCE_DTYPES[layer]
        assert str(array.dtype)==expected
        assert valid.dtype==bool and meta['crs']=='EPSG:4326'
    with pytest.raises(ValueError,match='shape/dtype'):
        b.decode(raw/'layers/S1Hand/India_1_S1Hand.tif','S1Hand',7)
    with pytest.raises(ValueError,match='shape/dtype'):
        b.decode(raw/'layers/LabelHand/India_1_LabelHand.tif','JRCWaterHand',8)
    _,valid,_=b.decode(raw/'layers/LabelHand/India_1_LabelHand.tif','LabelHand',8)
    assert not valid[0,0,0]


def test_alignment_tolerance_and_exact_crs_shape(raw):
    meta={layer:b.decode(raw/'layers'/layer/f'India_1_{layer}.tif',layer,8)[2] for layer in b.LAYERS}
    baseline=meta['S1Hand']['transform'][0]
    rounded=copy.deepcopy(meta)
    rounded['LabelHand']['transform'][2]+=baseline*5e-7
    rounded['LabelHand']['transform'][0]+=baseline*5e-7
    result=b.assert_alignment(rounded)
    assert result['max_origin_difference_pixels']==pytest.approx(5e-7,rel=.01)
    assert result['max_pixel_size_difference_pixels']==pytest.approx(5e-7,rel=.01)
    for coefficient in (0,2):
        bad=copy.deepcopy(meta)
        bad['LabelHand']['transform'][coefficient]+=baseline*2e-6
        with pytest.raises(ValueError,match='exceeds 1e-6'):
            b.assert_alignment(bad)
    for key,value in [('crs','EPSG:3857'),('shape',[7,8])]:
        bad=copy.deepcopy(meta); bad['JRCWaterHand'][key]=value
        with pytest.raises(ValueError,match='alignment mismatch'):
            b.assert_alignment(bad)


def test_pairing_check_catches_swapped_chip(pack):
    output,manifest=pack
    path=output/'train/s1.npy'
    array=np.load(path,mmap_mode='r+')
    array[[0,1]]=array[[1,0]]
    array.flush()
    with pytest.raises(ValueError,match='pairing/hash mismatch'):
        b.verify_pairing(output,manifest['chips'],['India_1'])


def test_split_disjointness_and_order(raw):
    splits=b.official_splits(raw)
    assert splits['train']==['India_1','India_2']
    with pytest.raises(ValueError,match='chip overlap'):
        b.assert_disjoint({'train':['a'],'test':['a']})
    with pytest.raises(ValueError,match='chip overlap'):
        b.assert_disjoint({'train':['a','a']})


def test_norm_stats_use_only_train_and_ignore_nodata(pack):
    _,manifest=pack
    stats=manifest['norm_stats']
    assert stats['s1']['mean']==[3,3]
    assert stats['s1']['std']==[1,1]
    assert stats['s1']['count']==[126,126]
    assert stats['s2']['mean']==[100]*13
    assert stats['s2']['count']==[126]*13
    assert all(s['split']=='train' for s in stats.values())


def test_manifest_schema_quality_and_full_verification(pack):
    output,m=pack
    assert {'pack_id','version','created','builder_commit','files','counts','band_order','units',
            'norm_stats','checks','licence','sources','chips','redistribution'}<=m.keys()
    assert m['redistribution']=='not allowed until licence is confirmed'
    assert m['licence']['status']=={'source':'proprietary','label':'proprietary'}
    assert m['licence']['reuse_terms']=='unknown'
    assert len(m['files'])==22 and len(m['chips'])==5
    assert m['counts']['per_split']=={'train':2,'valid':1,'test':1,'bolivia':1}
    assert m['counts']['india_per_split']=={'train':2,'valid':0,'test':1,'bolivia':0}
    assert not m['checks']['resampled']
    import pandas as pd
    index=pd.read_parquet(output/'index.parquet')
    assert {'chip_id','split','event','is_india','nodata_pct','water_pct','flags',
            'max_origin_difference_pixels','max_pixel_size_difference_pixels'}<=set(index)
    chip=m['chips']['Ghana_3']
    assert chip['quality']['s2_int16_max_pixels']==1
    assert chip['quality']['s2_ge_10000_pixels']==63
    assert sum(chip['quality']['s2_int16_max_per_band'])==1
    assert chip['quality']['s2_saturation_count']=='unknown'
    assert chip['quality']['water_pct']==pytest.approx(16/63*100)
    assert chip['quality']['nodata_pct']==pytest.approx(1/64*100)
    assert b.verify_pack(output)['pack_id']==m['pack_id']
    with (output/'valid/jrc.npy').open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError,match='file hash mismatch'):
        b.verify_pack(output)


def test_bad_label_encoding_stops(raw):
    path=raw/'layers/LabelHand/India_1_LabelHand.tif'
    with rasterio.open(path,'r+') as ds:
        array=ds.read();array[0,0,1]=7;ds.write(array)
    with pytest.raises(ValueError,match='unexpected label values'):
        b.decode(path,'LabelHand',8)


def test_high_nodata_is_flagged_not_dropped(raw,tmp_path):
    path=raw/'layers/LabelHand/India_1_LabelHand.tif'
    with rasterio.open(path,'r+') as ds:
        array=ds.read();array[:,:,:5]=-1;ds.write(array)
    log=json.loads((raw/'FETCH_LOG.json').read_text())
    entry=log[str(path.relative_to(raw))];entry['sha256']=b.sha256(path);entry['bytes']=path.stat().st_size
    (raw/'FETCH_LOG.json').write_text(json.dumps(log))
    m=b.assemble(raw,tmp_path/'flagged','a'*40,8)
    assert m['flagged_chips']==['India_1']
    assert len(m['chips'])==5 and m['counts']['per_split']['train']==2


def test_fetch_logs_hash_checks_cache_and_stops_on_access_failure(tmp_path,monkeypatch):
    import io
    calls=[]
    def response(url,timeout):
        calls.append(url)
        return io.BytesIO(b'synthetic downloaded bytes')
    monkeypatch.setattr(b.urllib.request,'urlopen',response)
    root=tmp_path/'fetched'
    p=b.fetch(root,'test.txt','https://synthetic.invalid/test')
    log=json.loads((root/'FETCH_LOG.json').read_text())
    assert log['test.txt']['sha256']==b.sha256(p)
    assert log['test.txt']['url']=='https://synthetic.invalid/test'
    b.fetch(root,'test.txt','https://synthetic.invalid/test')
    assert len(calls)==1
    p.write_text('tampered')
    with pytest.raises(ValueError,match='cached fetch mismatch'):
        b.fetch(root,'test.txt','https://synthetic.invalid/test')
    def denied(*args,**kwargs):
        raise OSError('access denied')
    monkeypatch.setattr(b.urllib.request,'urlopen',denied)
    with pytest.raises(RuntimeError,match='STOP: anonymous fetch failed'):
        b.fetch(root,'denied.txt','https://synthetic.invalid/denied')
