"""Build only the official Sen1Floods11 hand-labeled flood pack (no training)."""
import hashlib
import json
from pathlib import Path
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import xml.etree.ElementTree as ET

FETCH_LOCK = threading.Lock()

BUCKET = 'https://storage.googleapis.com/sen1floods11/'
LAYERS = ('S1Hand', 'S2Hand', 'LabelHand', 'S1OtsuLabelHand', 'JRCWaterHand')
SPLITS = {'train':'flood_train_data.csv', 'valid':'flood_valid_data.csv',
          'test':'flood_test_data.csv', 'bolivia':'flood_bolivia_data.csv'}


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(raw, name, url):
    """Anonymous HTTPS only; fail immediately on access errors. Never trust cached bytes."""
    raw = Path(raw)
    raw.mkdir(parents=True,exist_ok=True)
    logpath = raw/'FETCH_LOG.json'
    log = json.loads(logpath.read_text()) if logpath.exists() else {}
    path = raw/name
    if name in log:
        if not path.is_file() or sha256(path) != log[name]['sha256'] or log[name]['url'] != url:
            raise ValueError(f'STOP: cached fetch mismatch: {name}')
        return path
    if path.exists():
        raise ValueError(f'STOP: unlogged raw file exists: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.part')
    try:
        with urllib.request.urlopen(url,timeout=60) as response, tmp.open('wb') as target:
            for block in iter(lambda:response.read(1 << 20),b''):
                target.write(block)
        tmp.rename(path)
    except Exception as exc:
        raise RuntimeError(f'STOP: anonymous fetch failed for {url}: {exc}') from exc
    with FETCH_LOCK:
        log = json.loads(logpath.read_text()) if logpath.exists() else {}
        log[name] = {'url':url,'sha256':sha256(path),'bytes':path.stat().st_size}
        temporary = logpath.with_suffix('.tmp')
        temporary.write_text(json.dumps(log,indent=2,sort_keys=True)+'\n')
        temporary.replace(logpath)
    return path


def official_splits(raw):
    result = {}
    for split, filename in SPLITS.items():
        rows = list(csv.reader((Path(raw)/'splits'/filename).open()))
        ids = []
        for row in rows:
            if len(row) != 2 or not row[0].endswith('_S1Hand.tif'):
                raise ValueError(f'STOP: unexpected official split row: {row}')
            chip = row[0].removesuffix('_S1Hand.tif')
            if row[1] != chip+'_LabelHand.tif':
                raise ValueError('STOP: official pairing mismatch')
            ids.append(chip)
        result[split] = ids
    assert_disjoint(result)
    return result


def assert_disjoint(splits):
    seen = set()
    for split, ids in splits.items():
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'STOP: chip overlap in {split}')
        seen.update(ids)


def inventory(raw):
    ns = {'b':'http://doc.s3.amazonaws.com/2006-03-01'}
    listings = {}
    for layer in LAYERS:
        prefix = 'v1.1/data/flood_events/HandLabeled/'+layer+'/'
        path = fetch(raw,'sources/list-'+layer+'.xml',BUCKET+'?prefix='+prefix)
        root = ET.fromstring(path.read_bytes())
        if root.find('b:IsTruncated',ns).text != 'false':
            raise ValueError('STOP: bucket inventory truncated')
        rows = {}
        for x in root.findall('b:Contents',ns):
            key = x.find('b:Key',ns).text
            name = Path(key).name
            if not name.endswith('_'+layer+'.tif'):
                raise ValueError(f'STOP: unexpected hand-layer object {key}')
            chip = name.removesuffix('_'+layer+'.tif')
            rows[chip] = {'key':key,'bytes':int(x.find('b:Size',ns).text)}
        listings[layer] = rows
    expected = set(listings[LAYERS[0]])
    if any(set(rows) != expected for rows in listings.values()):
        raise ValueError('STOP: hand-labeled layer chip sets differ')
    splits = official_splits(raw)
    if set(chip for chips in splits.values() for chip in chips) != expected:
        raise ValueError('STOP: official splits do not cover exactly the hand-labeled inventory')
    return listings


def download_layers(raw, workers=6):
    listings = inventory(raw)
    jobs = [(layer, chip, info) for layer, chips in listings.items() for chip,info in chips.items()]
    def one(job):
        layer, chip, info = job
        path = fetch(raw,'layers/'+layer+'/'+chip+'_'+layer+'.tif',BUCKET+info['key'])
        if path.stat().st_size != info['bytes']:
            raise ValueError(f'STOP: download size mismatch: {path}')
        return path
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(one,job) for job in jobs]
    try:
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i % 50 == 0 or i == len(jobs):
                print(f'Fetched/verified {i}/{len(jobs)} hand-layer files',flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return listings


# Observed on the public v1.1 sample; unexpected source encodings stop decoding.
SOURCE_DTYPES = {'S1Hand':'float32', 'S2Hand':'int16', 'LabelHand':'int16',
                 'S1OtsuLabelHand':'int16', 'JRCWaterHand':'uint8'}
BANDS = {'S1Hand':['VV','VH'],
         'S2Hand':['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']}
CANONICAL_S2 = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
OUTPUTS = {'S1Hand':'s1', 'S2Hand':'s2', 'LabelHand':'label', 'S1OtsuLabelHand':'otsu', 'JRCWaterHand':'jrc'}


def array_hash(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def decode(path, layer, chip_size=512):
    import numpy as np
    import rasterio
    with rasterio.open(path) as ds:
        array = ds.read()
        bands = len(BANDS.get(layer,[None]))
        if array.shape != (bands,chip_size,chip_size) or str(array.dtype) != SOURCE_DTYPES[layer]:
            raise ValueError(f'STOP: shape/dtype mismatch in {path}: {array.shape}, {array.dtype}')
        if layer in BANDS and list(ds.descriptions) != BANDS[layer]:
            raise ValueError(f'STOP: unverified band order in {path}: {ds.descriptions}')
        if ds.crs is None:
            raise ValueError(f'STOP: missing CRS in {path}')
        valid = (ds.read_masks() != 0) & np.isfinite(array)
        if layer in ('LabelHand','S1OtsuLabelHand'):
            values = set(np.unique(array).tolist())
            if not values <= {-1,0,1}:
                raise ValueError(f'STOP: unexpected label values in {path}: {values}')
            valid &= array != -1
            array = array.astype(np.int8)
        meta = {'crs':ds.crs.to_string(),'transform':list(ds.transform),
                'shape':list(array.shape[-2:]),'source_dtype':ds.dtypes[0],
                'pack_dtype':str(array.dtype),'nodata':('NaN' if ds.nodata is not None and np.isnan(ds.nodata) else ds.nodata),
                'descriptions':list(ds.descriptions),'scales':list(ds.scales),'offsets':list(ds.offsets),
                'pixel_size_native':list(ds.res),'tags':ds.tags(),
                'band_tags':[ds.tags(i) for i in ds.indexes],
                'source_values':np.unique(array).tolist() if layer not in BANDS else None}
    return array, valid, meta


def assert_alignment(metadata):
    """Compare native affine transforms in reference S1 pixel coordinates; never resample."""
    import numpy as np
    base = metadata['S1Hand']
    t = base['transform']
    basis = np.array([[t[0],t[1]],[t[3],t[4]]],dtype=float)
    maxima = {'max_origin_difference_pixels':0., 'max_pixel_size_difference_pixels':0.}
    for layer,m in metadata.items():
        for key in ('crs','shape'):
            if m[key] != base[key]:
                raise ValueError(f'STOP: alignment mismatch: {layer} {key}')
        v = m['transform']
        origin = np.linalg.solve(basis,np.array([v[2]-t[2],v[5]-t[5]]))
        step = np.linalg.solve(basis,np.array([[v[0]-t[0],v[1]-t[1]],[v[3]-t[3],v[4]-t[4]]]))
        differences = {'max_origin_difference_pixels':float(np.max(np.abs(origin))),
                       'max_pixel_size_difference_pixels':float(np.max(np.abs(step)))}
        if any(not np.isfinite(x) or x > 1e-6 for x in differences.values()):
            raise ValueError(f'STOP: alignment transform exceeds 1e-6 pixel: {layer}: {differences}')
        for key,value in differences.items():
            maxima[key] = max(maxima[key],value)
    return maxima


def approximate_gsd(meta):
    """Local UTM distances at chip center; native angular resolution is also preserved."""
    import math
    from affine import Affine
    from rasterio.warp import transform
    a = Affine(*meta['transform'][:6])
    h,w = meta['shape']
    points = [a*(w/2,h/2),a*(w/2+1,h/2),a*(w/2,h/2+1)]
    xs,ys = zip(*points)
    lon,lat = transform(meta['crs'],'EPSG:4326',[xs[0]],[ys[0]])
    if not -80 <= lat[0] <= 84:
        return 'unknown'
    zone = min(60,max(1,int((lon[0]+180)//6)+1))
    utm = (32600 if lat[0] >= 0 else 32700)+zone
    xx,yy = transform(meta['crs'],f'EPSG:{utm}',xs,ys)
    return [math.hypot(xx[i]-xx[0],yy[i]-yy[0]) for i in (1,2)]


def update_stats(state, arrays, valid):
    """Streaming per-band moments; caller must pass TRAIN chips only."""
    import numpy as np
    for name in ('S1Hand','S2Hand'):
        array, mask = arrays[name], valid[name]
        if name not in state:
            state[name] = {'sum':np.zeros(len(array)), 'sum_sq':np.zeros(len(array)),
                           'count':np.zeros(len(array),dtype=np.int64)}
        s = state[name]
        for band in range(len(array)):
            x = array[band][mask[band]].astype(np.float64)
            s['sum'][band] += x.sum()
            s['sum_sq'][band] += np.square(x).sum()
            s['count'][band] += len(x)


def finish_stats(state):
    import numpy as np
    result = {}
    for layer,s in state.items():
        if np.any(s['count'] == 0):
            raise ValueError(f'STOP: no valid train pixels in {layer}')
        mean = s['sum']/s['count']
        variance = np.maximum(s['sum_sq']/s['count']-mean**2,0)
        result[OUTPUTS[layer]] = {'count':s['count'].tolist(),'mean':mean.tolist(),
                                 'std':np.sqrt(variance).tolist(),
                                 'units':'dB' if layer == 'S1Hand' else 'TOA reflectance DN (divide by 10000)',
                                 'split':'train','mask':'per-band source raster mask AND finite; no label-based filtering'}
    return result


def verify_pairing(pack, chips, sample_ids):
    """Re-open every sampled chip in all five arrays; hashes are keyed to chip ID."""
    import numpy as np
    pack = Path(pack)
    arrays = {}
    for chip in sample_ids:
        record = chips[chip]
        for layer,name in OUTPUTS.items():
            file = record['split']+'/'+name+'.npy'
            if file not in arrays:
                arrays[file] = np.load(pack/file,mmap_mode='r')
            actual = array_hash(arrays[file][record['row']])
            if actual != record['array_sha256'][layer]:
                raise ValueError(f'STOP: chip pairing/hash mismatch: {chip} {layer}')
    return True


def verify_pack(pack):
    import numpy as np
    import pandas as pd
    pack = Path(pack)
    manifest = json.loads((pack/'MANIFEST.json').read_text())
    actual_files = {str(p.relative_to(pack)) for p in pack.rglob('*') if p.is_file() and p != pack/'MANIFEST.json'}
    if actual_files != set(manifest['files']):
        raise ValueError('STOP: manifest file inventory mismatch')
    for name, info in manifest['files'].items():
        if sha256(pack/name) != info['sha256'] or (pack/name).stat().st_size != info['bytes']:
            raise ValueError(f'STOP: pack file hash mismatch: {name}')
    index = pd.read_parquet(pack/'index.parquet')
    if index.chip_id.duplicated().any() or set(index.chip_id) != set(manifest['chips']):
        raise ValueError('STOP: index chip identity mismatch')
    for row in index.itertuples():
        c = manifest['chips'][row.chip_id]
        if row.split != c['split'] or row.row != c['row']:
            raise ValueError('STOP: index row pairing mismatch')
        measured = assert_alignment(c['metadata'])
        for key,value in measured.items():
            if c['alignment'][key] != value or getattr(row,key) != value:
                raise ValueError('STOP: index/manifest alignment measurement mismatch')
    assert_disjoint({s:index.loc[index.split==s,'chip_id'].tolist() for s in SPLITS})
    for split,n in manifest['counts']['per_split'].items():
        if int((index.split == split).sum()) != n:
            raise ValueError('STOP: split count mismatch')
        for layer,name in OUTPUTS.items():
            a = np.load(pack/split/(name+'.npy'),mmap_mode='r')
            bands = len(BANDS.get(layer,[None]))
            if list(a.shape) != [n,bands,*manifest['chip_size']] or str(a.dtype) != manifest['dtypes'][name]:
                raise ValueError(f'STOP: packed shape/dtype mismatch in {split}/{name}')
    # Stronger than the required random 5%: real-pack verification checks every chip.
    verify_pairing(pack,manifest['chips'],list(manifest['chips']))
    return manifest


def assemble(raw, pack, builder_commit, chip_size=512):
    import datetime
    import math
    import shutil
    from collections import Counter
    import numpy as np
    import pandas as pd
    raw, pack = Path(raw),Path(pack)
    if pack.exists():
        raise ValueError(f'STOP: pack destination already exists: {pack}')
    log = json.loads((raw/'FETCH_LOG.json').read_text())
    for name,info in log.items():
        if sha256(raw/name) != info['sha256']:
            raise ValueError(f'STOP: raw hash mismatch: {name}')
    splits = official_splits(raw)
    pack.mkdir(parents=True)
    records, chips, state = [], {}, {}
    for split, ids in splits.items():
        folder = pack/split
        folder.mkdir()
        out = {}
        for layer,name in OUTPUTS.items():
            dtype = 'int8' if layer in ('LabelHand','S1OtsuLabelHand') else SOURCE_DTYPES[layer]
            out[layer] = np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=dtype,
                            shape=(len(ids),len(BANDS.get(layer,[None])),chip_size,chip_size))
        for row, chip in enumerate(ids):
            arrays, valid, metadata = {}, {}, {}
            for layer in LAYERS:
                name = 'layers/'+layer+'/'+chip+'_'+layer+'.tif'
                if name not in log:
                    raise ValueError(f'STOP: source file absent from fetch log: {name}')
                arrays[layer],valid[layer],metadata[layer] = decode(raw/name,layer,chip_size)
            alignment = assert_alignment(metadata)
            invalid = np.zeros((chip_size,chip_size),dtype=bool)
            nodata_by_layer = {}
            for layer in LAYERS:
                bad = ~valid[layer].all(0)
                invalid |= bad
                nodata_by_layer[layer] = float(bad.mean()*100)
                out[layer][row] = arrays[layer]
            label = arrays['LabelHand'][0]
            label_valid = valid['LabelHand'][0]
            water_pct = float((label[label_valid] == 1).mean()*100) if label_valid.any() else None
            nodata_pct = float(invalid.mean()*100)
            flags = ['nodata_gt_50pct'] if nodata_pct > 50 else []
            s1 = arrays['S1Hand']
            s2_at_max = arrays['S2Hand'] == 32767
            s2_ge_10000 = arrays['S2Hand'] >= 10000
            quality = {'nodata_pct':nodata_pct,'nodata_pct_by_layer':nodata_by_layer,
                       'label_nodata_pct':nodata_by_layer['LabelHand'],'water_pct':water_pct,
                       'label_water_pixels':int(((label == 1)&label_valid).sum()),
                       'label_valid_pixels':int(label_valid.sum()),'s1_nan_count':int(np.isnan(s1).sum()),
                       's1_inf_count':int(np.isinf(s1).sum()),'s2_saturation_count':'unknown',
                       's2_int16_max_pixels':int(s2_at_max.any(0).sum()),
                       's2_ge_10000_pixels':int(s2_ge_10000.any(0).sum()),
                       's2_int16_max_per_band':s2_at_max.sum((1,2)).tolist(),
                       's2_ge_10000_per_band':s2_ge_10000.sum((1,2)).tolist(),
                       's2_raw_count_physical_meaning':'unknown'}
            chips[chip] = {'split':split,'row':row,'event':chip.rsplit('_',1)[0],
                           'is_india':chip.startswith('India_'),'flags':flags,'quality':quality,
                           'metadata':metadata,'alignment':alignment,'gsd_approx_m':approximate_gsd(metadata['S1Hand']),
                           'array_sha256':{layer:array_hash(a) for layer,a in arrays.items()}}
            records.append({'chip_id':chip,'split':split,'row':row,'event':chips[chip]['event'],
                            'is_india':chips[chip]['is_india'],'nodata_pct':nodata_pct,
                            'water_pct':water_pct,'flags':flags,'s1_nan_count':quality['s1_nan_count'],
                            's1_inf_count':quality['s1_inf_count'],'s2_saturation_count':'unknown',
                            's2_int16_max_pixels':quality['s2_int16_max_pixels'],
                            's2_ge_10000_pixels':quality['s2_ge_10000_pixels'],**alignment})
            if split == 'train':
                update_stats(state,arrays,valid)
        for a in out.values():
            a.flush()
        del out
        print(f'Packed {split}: {len(ids)} chips',flush=True)
    pd.DataFrame(records).to_parquet(pack/'index.parquet',index=False)
    shutil.copyfile(raw/'FETCH_LOG.json',pack/'FETCH_LOG.json')
    rng = np.random.default_rng(17)
    sample = sorted(rng.choice(sorted(chips),size=max(1,math.ceil(.05*len(chips))),replace=False).tolist())
    verify_pairing(pack,chips,sample)
    licence = {k:json.loads((raw/'sources'/('hand-'+k+'-collection.json')).read_text()).get('license','unknown') for k in ('source','label')}
    counts = {'per_split':{s:len(ids) for s,ids in splits.items()},
              'per_event':dict(sorted(Counter(c['event'] for c in chips.values()).items())),
              'india_per_split':{s:sum(c.startswith('India_') for c in ids) for s,ids in splits.items()},
              'per_split_event':{s:dict(sorted(Counter(c.rsplit('_',1)[0] for c in ids).items())) for s,ids in splits.items()}}
    water = np.array([c['quality']['water_pct'] for c in chips.values() if c['quality']['water_pct'] is not None])
    total_valid = sum(c['quality']['label_valid_pixels'] for c in chips.values())
    total_water = sum(c['quality']['label_water_pixels'] for c in chips.values())
    manifest = {'pack_id':'sen1floods11-hand-v1','version':1,
        'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'builder_commit':builder_commit,
        'builder_sha256':sha256(Path(__file__)),'chip_size':[chip_size,chip_size],
        'band_order':{'s1':['VV','VH'],'s2':CANONICAL_S2},
        'units':{'s1':{'value':'dB','basis':'official README; source VV/VH descriptions verified; raster value-unit tag absent'},
                 's2':{'value':'TOA reflectance DN','physical_scale':.0001,'product_level':'L1C','basis':'official README; source band descriptions verified'}},
        'dtypes':{'s1':'float32','s2':'int16','label':'int8','otsu':'int8','jrc':'uint8'},
        'label_encoding':{'-1':'nodata / invalid','0':'no water','1':'water'},
        'redistribution':'not allowed until licence is confirmed',
        'licence':{'status':licence,'reuse_terms':'unknown','flag':'STAC source and label collections declare proprietary; README supplies no reuse licence; GitHub API license is null'},
        'sources':{'repository':'https://github.com/cloudtostreet/Sen1Floods11','bucket':BUCKET,
                   'fetch_log':'FETCH_LOG.json','all_download_urls_and_sha256':'FETCH_LOG.json',
                   'scope':'Only data/flood_events/HandLabeled; no weak labels or permanent-water dataset'},
        'gsd':{'readme_nominal_m':10,'native_units':'per-chip CRS units; raster transform stored per layer',
               'approx_m_method':'Adjacent pixel-center distance projected to local UTM at chip center; not a uniform 10 m ground grid'},
        'differences_from_readme':['Observed S2 source dtype Int16, not README UInt16; preserved without unsigned conversion',
                                   'LabelHand header nodata is unset; explicit -1 values are treated as nodata'],
        'quality_definitions':{'nodata_pct':'union of any invalid band across all five layers, percent of chip pixels',
            'water_pct':'water pixels / valid LabelHand pixels * 100; excludes -1',
            's2_saturation_count':'unknown',
            's2_int16_max_pixels':'raw spatial pixels with any band ==32767; unknown physical meaning',
            's2_ge_10000_pixels':'raw spatial pixels with any band >=10000; unknown physical meaning',
            's2_per_band_counts':'raw band samples, separately per band; unknown physical meaning'},
        'norm_stats':finish_stats(state),'counts':counts,
        'water_pct_summary':{'chips_with_valid_labels':len(water),'min':float(water.min()) if len(water) else None,
            'median':float(np.median(water)) if len(water) else None,'mean':float(water.mean()) if len(water) else None,
            'max':float(water.max()) if len(water) else None,'pooled_valid_pixel_water_pct':100*total_water/total_valid if total_valid else None},
        'flagged_chips':[k for k,c in chips.items() if c['flags']],
        'checks':{'split_overlap':0,'official_order_preserved':True,'aligned_layers':list(LAYERS),
                  'reopened_pairing_fraction':.05,'reopened_pairing_count':len(sample),'reopened_chip_ids':sample,'seed':17,
                  'train_only_norm_stats':True,'no_chips_dropped':True,
                  'alignment_tolerance_pixels':1e-6,'resampled':False},
        'chips':chips,'files':{}}
    for path in sorted(pack.rglob('*')):
        if path.is_file():
            manifest['files'][str(path.relative_to(pack))] = {'sha256':sha256(path),'bytes':path.stat().st_size}
    (pack/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def fetch_sources(raw):
    for name,url in [('README.md','https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/README.md'),
                     ('docs-README.md','https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/docs/README.md'),
                     ('github-repository.json','https://api.github.com/repos/cloudtostreet/Sen1Floods11'),
                     ('catalog.json',BUCKET+'v1.1/catalog/catalog.json'),
                     ('event-metadata.geojson',BUCKET+'v1.1/Sen1Floods11_Metadata.geojson')]:
        fetch(raw,'sources/'+name,url)
    for kind in ('source','label'):
        fetch(raw,'sources/hand-'+kind+'-collection.json',BUCKET+'v1.1/catalog/sen1floods11_hand_labeled_'+kind+'/collection.json')
    for filename in SPLITS.values():
        fetch(raw,'splits/'+filename,BUCKET+'v1.1/splits/flood_handlabeled/'+filename)


def main():
    import argparse
    import subprocess
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path.home()/'satquery-data/flood/raw')
    p.add_argument('--pack',type=Path,default=Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1')
    p.add_argument('--verify-only',action='store_true')
    args = p.parse_args()
    if args.verify_only:
        verify_pack(args.pack)
        print('All pack file hashes, index identity, arrays and per-chip pairing: PASS')
    else:
        fetch_sources(args.raw)
        download_layers(args.raw)
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
        dirty = subprocess.check_output(['git','status','--porcelain','--',str(Path(__file__).resolve())],cwd=Path(__file__).parent,text=True).strip()
        if dirty:
            raise ValueError('STOP: commit builder before assembling pack for truthful builder_commit')
        assemble(args.raw,args.pack,commit)
    print('MANIFEST sha256:',sha256(args.pack/'MANIFEST.json'))


if __name__ == '__main__':
    main()
