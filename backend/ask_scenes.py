"""Trusted local pack references and strictly described GeoTIFF uploads for /ask."""
import base64
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from PIL import Image
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds, Affine

from backend.fresh_scenes import BACKUPS
from backend.services import (InvalidImageUpload, SceneNotFound, MAX_RASTER_BANDS,
                              MAX_RASTER_PIXELS, _sha256_file, _preview_channel)
from core.schema import Scene, scene_id_for
from core.tools.landcover import BANDS, UNITS

UPLOAD_ID = re.compile(r'upload_[0-9a-f]{32}')
OPTICAL = {'B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'}
KNOWN_BANDS = OPTICAL | {'VV','VH','dataMask','red','green','blue'}


@dataclass
class LoadedScene:
    scene: Scene
    arrays: dict
    transform: Affine
    context: Any = None
    native: Path | None = None
    provenance: dict | None = None

    def means(self):
        return {b: float(x[np.isfinite(x)].mean(dtype=np.float64))
                for item in self.arrays.values() for b,x in zip(item['bands'],item['pixels'])
                if np.isfinite(x).any()}

    def pixels(self, tool, mode, directory):
        if tool == 'landcover':
            keys=('s1','s2') if mode=='S1S2' else (mode.lower(),)
            return {key:self.arrays[key] for key in keys}
        if tool == 'flood-otsu':
            return self.context() if self.context else {'s1':self.arrays['s1']}
        if tool=='change-deterministic' and mode=='S1':
            raise ValueError('Legacy S1 change provider requires linear backscatter, not dB')
        if self.native:
            return str(self.native)
        # The existing VLM/DINO adapters consume display images, not spectral tensors.
        if 's2' not in self.arrays: raise ValueError('This tool requires an optical display image')
        target=directory/(self.scene.scene_id+'.png');directory.mkdir(parents=True,exist_ok=True)
        if not target.exists(): self.preview_image().save(target)
        return str(target)

    def preview_image(self):
        if 's2' in self.arrays:
            item, names = self.arrays['s2'], ('B04', 'B03', 'B02')
        elif 's1' in self.arrays:
            item, names = self.arrays['s1'], ('VV',)
        else:
            item, names = self.arrays['rgb'], ('red', 'green', 'blue')
        if any(name not in item['bands'] for name in names):
            raise ValueError('Scene does not contain the preview bands')
        channels = [_preview_channel(np.ma.masked_invalid(item['pixels'][item['bands'].index(name)]))
                    for name in names]
        return Image.fromarray(channels[0] if len(channels) == 1 else np.stack(channels, -1))

    def preview(self):
        from io import BytesIO

        image = self.preview_image()
        image.thumbnail((320, 320))
        with BytesIO() as stream:
            image.save(stream, format='PNG')
            url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')
        return {'image': url, 'scene': self.scene, 'provenance': self.provenance,
                'preview_kind': 'VV grayscale' if image.mode == 'L' else 'RGB quicklook'}


def make_scene(arrays, *, crs, transform, source, cloud=None, invalid=None, metadata=None, extra_bands=(), acquired=None):
    keys=tuple(k for k in ('s1','s2','rgb') if k in arrays)
    sensor='S1S2' if keys==('s1','s2') else {'s1':'S1','s2':'S2','rgb':'RGB_HR'}[keys[0]]
    first=arrays[keys[0]]['pixels'];height,width=first.shape[-2:]
    if any(v['pixels'].shape[-2:]!=(height,width) for v in arrays.values()):
        raise ValueError('Sensor grids differ')
    bounds=array_bounds(height,width,transform)
    crs_obj=rasterio.crs.CRS.from_user_input(crs)
    if crs_obj.is_projected:
        gsd=math.hypot(transform.a,transform.d)*crs_obj.linear_units_factor[1]
    else:
        from rasterio.warp import transform as warp
        x,y=transform*(width/2,height/2);x2,y2=transform*(width/2+1,height/2)
        lon,lat=warp(crs_obj,'EPSG:4326',[x,x2],[y,y2])
        a,b=map(math.radians,lat);dl=math.radians(lon[1]-lon[0])
        gsd=6371008.8*2*math.asin(min(1,math.sqrt(math.sin((b-a)/2)**2+math.cos(a)*math.cos(b)*math.sin(dl/2)**2)))
    if invalid is None:
        invalid=np.zeros((height,width),dtype=bool)
        for item in arrays.values(): invalid |= ~np.isfinite(item['pixels']).all(0)
    info=dict(sensor=sensor,bands=tuple(b for k in keys for b in arrays[k]['bands'])+tuple(extra_bands),
              units='unknown' if len(keys)>1 else arrays[keys[0]]['units'],gsd_m=gsd,
              crs=crs_obj.to_string(),bounds=tuple(bounds),source=source,
              cloud_pct=cloud,nodata_pct=float(invalid.mean()*100))
    identity={**info,'transform':list(transform),'shape':[height,width],'validity_sha256':hashlib.sha256(invalid.tobytes()).hexdigest(),**(metadata or {})}
    raw=b''.join(np.ascontiguousarray(arrays[k]['pixels']).tobytes() for k in keys)
    return Scene(scene_id=scene_id_for(raw,identity), acquired=acquired, **info)


class SceneStore:
    def __init__(self, config):
        self.config=config
        self.verified={}

    def verify(self,path,digest):
        path=Path(path)
        signature=(path.stat().st_mtime_ns,path.stat().st_size,digest)
        if self.verified.get(path)!=signature:
            if _sha256_file(path)!=digest: raise ValueError(f'Pack checksum mismatch: {path.name}')
            self.verified[path]=signature

    def raster(self,data, *, fresh_mode=None):
        try:
            with MemoryFile(data) as memory, memory.open() as src:
                if src.driver!='GTiff' or src.crs is None or src.transform==Affine.identity():
                    raise InvalidImageUpload('A GeoTIFF with CRS and affine georeferencing is required')
                if (src.count<1 or src.count>MAX_RASTER_BANDS or src.width*src.height>MAX_RASTER_PIXELS
                    or sum(np.dtype(t).itemsize for t in src.dtypes)*src.width*src.height>self.config.decoded_limit):
                    raise InvalidImageUpload('Decoded raster exceeds safety limits (512 MiB)')
                names=tuple(src.descriptions)
                if any(b not in KNOWN_BANDS for b in names) or len(set(names))!=len(names):
                    raise InvalidImageUpload('Each band needs a unique supported description (e.g. VV, VH or B02)')
                if any(np.dtype(t).kind not in 'uif' for t in src.dtypes):
                    raise InvalidImageUpload('Unsupported raster numeric type')
                pixels=src.read();invalid=~np.isfinite(pixels).all(0)
                for i in range(src.count): invalid |= src.read_masks(i+1)==0
                if 'dataMask' in names:
                    mask=pixels[names.index('dataMask')]
                    valid_mask=src.read_masks(names.index('dataMask')+1)>0
                    if not np.isin(mask[valid_mask],[0,1]).all():
                        raise InvalidImageUpload('dataMask must contain 0 or 1 at valid pixels')
                    invalid |= mask==0
                if invalid.all() and fresh_mode is None: raise InvalidImageUpload('Raster contains no valid pixels')
                tags=src.tags();arrays={}
                for key,allowed in [('s1',{'VV','VH'}),('s2',OPTICAL),('rgb',{'red','green','blue'})]:
                    idx=[i for i,b in enumerate(names) if b in allowed]
                    if not idx: continue
                    units=UNITS.get(key,'uint8')
                    if any((src.units[i] or src.tags(i+1).get('units') or tags.get('units'))!=units for i in idx):
                        raise InvalidImageUpload(f'{key} bands require explicit units={units}')
                    arrays[key]={'pixels':pixels[idx], 'bands':tuple(names[i] for i in idx),'units':units}
                if not arrays or ('rgb' in arrays and len(arrays)>1):
                    raise InvalidImageUpload('Unsupported combination of sensor bands')
                cloud=float(tags['cloud_pct']) if 'cloud_pct' in tags else None
                provenance = None
                acquired = None
                source = 'uploaded GeoTIFF; cloud metadata declared by uploader'
                metadata = {'dataMask_sha256': hashlib.sha256(pixels[names.index('dataMask')].tobytes()).hexdigest()
                            if 'dataMask' in names else None}
                if fresh_mode:
                    provenance = json.loads(tags['satquery_provenance'])
                    provenance.update(mode=fresh_mode, label='Live scene' if fresh_mode == 'live' else 'Cached scene (offline)')
                    source = provenance['source']
                    acquired = datetime.fromisoformat(provenance['acquisition_datetime']).date()
                    metadata['provenance'] = provenance
                sc = make_scene(arrays, crs=src.crs, transform=src.transform, source=source,
                                cloud=cloud, invalid=invalid, acquired=acquired, metadata=metadata,
                                extra_bands=('dataMask',) if 'dataMask' in names else ())
                return LoadedScene(sc, arrays, src.transform, provenance=provenance)
        except InvalidImageUpload: raise
        except Exception as error:
            raise InvalidImageUpload(f'Cannot read GeoTIFF metadata/pixels ({type(error).__name__})') from error

    def upload(self,data):
        loaded=self.raster(data)
        upload_id='upload_'+uuid4().hex
        directory=self.config.data_dir/'uploads';directory.mkdir(parents=True,exist_ok=True)
        with (directory/(upload_id+'.tif')).open('xb') as stream: stream.write(data)
        with (directory/(upload_id+'.json')).open('x') as stream:
            json.dump({'scene':loaded.scene.model_dump(mode='json'),'sha256':hashlib.sha256(data).hexdigest()},stream)
        return {'upload_id':upload_id,'scene':loaded.scene}

    def save_fresh(self, data):
        digest = hashlib.sha256(data).hexdigest()
        directory = self.config.data_dir / 'fresh'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (digest + '.tif')
        with NamedTemporaryFile(dir=directory) as stream:
            stream.write(data)
            stream.flush()
            # Publish the content-addressed chip only after completing the write.
            try:
                os.link(stream.name, target)
            except FileExistsError:
                pass
        return digest

    def load(self,reference):
        if isinstance(reference,str):
            if not UPLOAD_ID.fullmatch(reference): raise SceneNotFound('Unknown upload id')
            path=self.config.data_dir/'uploads'/(reference+'.tif')
            info=path.with_suffix('.json')
            if not info.is_file() or not path.is_file(): raise SceneNotFound('Unknown upload id')
            meta=json.loads(info.read_text());self.verify(path,meta['sha256'])
            loaded=self.raster(path.read_bytes());loaded.native=path
            if loaded.scene!=Scene.model_validate(meta['scene']): raise ValueError('Upload metadata integrity mismatch')
            return loaded
        if reference.pack_id == 'fresh':
            digest = reference.patch_id
            if not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise SceneNotFound('Unknown fresh scene')
            path = self.config.data_dir / 'fresh' / (digest + '.tif')
            if not path.is_file():
                raise SceneNotFound('Unknown fresh scene')
            self.verify(path, digest)
            return self.raster(path.read_bytes(), fresh_mode='live')
        if reference.pack_id == 'fresh-backups':
            for row in json.loads((BACKUPS / 'manifest.json').read_text()):
                if row['id'] == reference.patch_id:
                    path = BACKUPS / (row['id'] + '.tif')
                    self.verify(path, row['sha256'])
                    return self.raster(path.read_bytes(), fresh_mode='cached')
            raise SceneNotFound('Unknown cached scene')
        if reference.pack_id=='worldcover': return self.worldcover(reference.patch_id)
        if reference.pack_id=='sen1floods11-hand-v1': return self.flood(reference.patch_id)
        raise SceneNotFound('Unknown pack id')

    def worldcover(self,patch_id):
        path=self.config.worldcover_dir
        if not (path/'MANIFEST.json').is_file(): raise SceneNotFound('WorldCover pack is absent')
        manifest=json.loads((path/'MANIFEST.json').read_text())
        entries={x['file']:x['sha256'] for x in manifest}
        for name in ('meta.parquet','classes.json','s1.npy','s2_10m.npy'): self.verify(path/name,entries[name])
        info=json.loads((path/'classes.json').read_text())
        if (tuple(info['s1_channels'])!=BANDS['s1'] or tuple(info['s2_channels'])!=BANDS['s2']
                or info['patch_px']!=120 or info['res_m']!=10
                or not info['s1_unit'].startswith('dB =')
                or not info['s2_unit'].startswith('DN = reflectance*10000')):
            raise ValueError('WorldCover band/order/unit/grid contract mismatch')
        import pandas as pd
        meta=pd.read_parquet(path/'meta.parquet')
        if not patch_id.isdigit(): raise SceneNotFound('WorldCover patch_id must be its numeric idx')
        matches=meta[meta.idx==int(patch_id)]
        if len(matches)!=1: raise SceneNotFound('Unknown WorldCover patch idx')
        row=matches.iloc[0];idx=int(row.idx)
        arrays={k:{'pixels':np.array(np.load(path/name,mmap_mode='r')[idx]),'bands':BANDS[k],'units':UNITS[k]}
                for k,name in [('s1','s1.npy'),('s2','s2_10m.npy')]}
        # build_wc.py stores patch center (offset=ul at PATCH/2), not a pixel center.
        transform=Affine(10,0,float(row.utm_x)-600,0,-10,float(row.utm_y)+600)
        digest=_sha256_file(path/'MANIFEST.json')
        sc=make_scene(arrays,crs=f'EPSG:{int(row.utm_epsg)}',transform=transform,source=f'worldcover:{idx}',
                      metadata={'manifest_sha256':digest})
        return LoadedScene(sc,arrays,transform)

    def flood(self,patch_id):
        path=self.config.flood_dir
        if not (path/'MANIFEST.json').is_file(): raise SceneNotFound('Sen1Floods11 pack is absent')
        manifest=json.loads((path/'MANIFEST.json').read_text())
        row=manifest['chips'].get(patch_id)
        if row is None: raise SceneNotFound('Unknown Sen1Floods11 chip id')
        if manifest['band_order']['s1']!=['VV','VH'] or manifest['units']['s1']['value']!='dB':
            raise ValueError('Pack S1 band/unit contract mismatch')
        def image(r):
            name=r['split']+'/s1.npy';self.verify(path/name,manifest['files'][name]['sha256'])
            return np.load(path/name,mmap_mode='r')[r['row']]
        raw=np.array(image(row));metadata=row['metadata']['S1Hand']
        arrays={'s1':{'pixels':raw,'bands':BANDS['s1'],'units':'dB'}}
        transform=Affine(*metadata['transform'][:6])
        sc=make_scene(arrays,crs=metadata['crs'],transform=transform,source=f'sen1floods11-hand-v1:{patch_id}',
                      metadata={'manifest_sha256':_sha256_file(path/'MANIFEST.json'),'event':row['event']})
        def context():
            members=sorted((name,r) for name,r in manifest['chips'].items() if r['event']==row['event'])
            stack=np.stack([image(r) for _,r in members])
            return {'s1':{'pixels':stack,'bands':BANDS['s1'],'units':'dB',
                          'chip_index':next(i for i,(name,_) in enumerate(members) if name==patch_id)}}
        return LoadedScene(sc,arrays,transform,context=context)
