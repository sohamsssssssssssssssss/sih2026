"""Public STAC acquisition and small, checksum-verified demonstration backups."""
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from time import monotonic
from typing import Annotated, Literal
from urllib.parse import urlparse

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform
from rasterio.windows import Window
from requests import RequestException

from backend.config import ROOT
from backend.geocode import GeocodeError, geocode
from core.tools.landcover import UNITS

BACKUPS = ROOT / "data" / "fresh-backups"
PROVIDERS = {
    "S2": ("Earth Search", "https://earth-search.aws.element84.com/v1", "sentinel-2-l2a"),
    "S1": ("Microsoft Planetary Computer", "https://planetarycomputer.microsoft.com/api/stac/v1", "sentinel-1-rtc"),
}
ASSETS = {
    "S2": {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"},
    "S1": {"VV": "vv", "VH": "vh"},
}
CHIP_SIZE = 120
SCENE_CLOUD_LABEL = "scene-level cloud cover"


class FreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    place: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    start_date: date
    end_date: date | None = None
    sensor: Literal["S1", "S2"] = "S2"

    @model_validator(mode="after")
    def valid_selection(self):
        has_point = self.lat is not None and self.lon is not None
        if (self.lat is None) != (self.lon is None) or bool(self.place) == has_point:
            raise ValueError("Supply a place or both latitude and longitude.")
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("End date must be on or after the start date.")
        return self


class ProviderError(RuntimeError):
    pass


class SceneUnavailable(LookupError):
    pass


def point_for(client, body):
    if body.place is None:
        return body.lon, body.lat
    candidates = geocode(client, body.place, india_only=False)
    if not candidates:
        raise SceneUnavailable("No location matched that place name. Try entering latitude and longitude.")
    return candidates[0].lon, candidates[0].lat


def _item_for(client, body, point):
    _, endpoint, collection = PROVIDERS[body.sensor]
    params = {
        "collections": collection,
        "intersects": json.dumps({"type": "Point", "coordinates": point}),
        "datetime": f"{body.start_date}T00:00:00Z/{body.end_date or body.start_date}T23:59:59.999999Z",
        "limit": 1,
        "sortby": "+properties.eo:cloud_cover" if body.sensor == "S2" else "-properties.datetime",
    }
    if body.sensor == "S1":
        params["query"] = json.dumps({"sar:polarizations": {"eq": ["VV", "VH"]}})
    response = client.get(f"{endpoint}/search", params=params)
    response.raise_for_status()
    items = response.json()["features"]
    if not items:
        raise SceneUnavailable("No matching scene was found for this location and date window.")
    return items[0]


def _needs_boa_offset(properties):
    applied = properties.get("earthsearch:boa_offset_applied", False)
    if applied is True:
        return False
    baseline = properties.get("s2:processing_baseline")
    if (applied is not False
            or not isinstance(baseline, str) or not re.fullmatch(r"[0-9]{2}\.[0-9]{2}", baseline)):
        raise ProviderError("Earth Search S2 BOA offset state is unknown: invalid or missing offset flag or processing baseline.")
    return int(baseline.split(".")[0]) >= 4


def _read_band(asset, point, sensor, needs_boa_offset=False):
    href = asset["href"]
    if sensor == "S1":
        from planetary_computer import sign

        href = sign(href)
    if urlparse(href).scheme != "https":
        raise ValueError("Expected an HTTPS COG asset")
    with rasterio.open("/vsicurl/" + href) as src:
        x, y = transform("EPSG:4326", src.crs, [point[0]], [point[1]])
        row, col = src.index(x[0], y[0])
        window = Window(col - CHIP_SIZE // 2, row - CHIP_SIZE // 2, CHIP_SIZE, CHIP_SIZE)
        pixels = src.read(1, masked=True, boundless=True, window=window, out_dtype="float32").filled(np.nan)
        grid = (src.crs, src.window_transform(window))
    if sensor == "S1":
        pixels[pixels <= 0] = np.nan
        pixels = 10 * np.log10(pixels)
    else:
        band = asset["raster:bands"][0]
        offset = band["offset"] if needs_boa_offset else 0
        pixels = (pixels * band["scale"] + offset) * 10000
    return pixels, grid


def fetch_chip(client, body, point, *, deadline=float("inf")):
    """Fetch only the 120 × 120 window from each aligned range-readable COG."""
    try:
        item = _item_for(client, body, point)
        name, _, collection = PROVIDERS[body.sensor]
        properties = item["properties"]
        acquired = datetime.fromisoformat(properties["datetime"].replace("Z", "+00:00"))
        provenance = {
            "source": name, "item_id": item["id"], "acquisition_datetime": acquired.isoformat(),
            "collection": collection, "mode": "live", "label": "Live scene",
        }
        cloud = float(properties["eo:cloud_cover"]) if body.sensor == "S2" else None
        if cloud is not None:
            if not 0 <= cloud <= 100:
                raise ValueError("Invalid eo:cloud_cover")
            provenance["cloud_pct_label"] = SCENE_CLOUD_LABEL
        assets = ASSETS[body.sensor]
        needs_boa_offset = body.sensor == "S2" and _needs_boa_offset(properties)
        # An S2 window can span multiple compressed 1024-pixel tiles in one range request.
        read_timeout = "60" if body.sensor == "S2" else "10"
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_CONNECTTIMEOUT="5",
                          GDAL_HTTP_TIMEOUT=read_timeout, GDAL_HTTP_MAX_RETRY="0"):
            reads = []
            for key in assets.values():
                if monotonic() >= deadline:
                    raise TimeoutError
                reads.append(_read_band(item["assets"][key], point, body.sensor, needs_boa_offset))
        crs, affine = reads[0][1]
        if any(grid != (crs, affine) for _, grid in reads):
            raise ValueError("STAC assets have different pixel grids")
        with MemoryFile() as buffer:
            with buffer.open(driver="GTiff", height=CHIP_SIZE, width=CHIP_SIZE, crs=crs,
                             transform=affine, count=len(assets), dtype="float32",
                             nodata=np.nan, compress="deflate") as dst:
                dst.write(np.stack([pixels for pixels, _ in reads]))
                dst.descriptions = tuple(assets)
                dst.units = (UNITS[body.sensor.lower()],) * len(assets)
                dst.update_tags(satquery_provenance=json.dumps(provenance))
                if cloud is not None:
                    dst.update_tags(cloud_pct=cloud)
            return buffer.read()
    except TimeoutError:
        raise
    except (httpx.HTTPError, RequestException, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise ProviderError(f"{PROVIDERS[body.sensor][0]} could not supply imagery ({type(error).__name__}).") from error


def backup_for(body, point):
    records = [row for row in json.loads((BACKUPS / "manifest.json").read_text()) if row["sensor"] == body.sensor]
    if point is None:
        # An offline place lookup can use a known demo name. Never guess an unknown place's region.
        name = body.place.casefold().strip()
        records = [row for row in records if row["place"].casefold() == name]
        if not records:
            raise SceneUnavailable("Place lookup is unavailable. Enter coordinates or a named demo location.")
        point = records[0]["point"]
    lon, lat = point
    row = min(records, key=lambda row: (
        (row["point"][0] - lon) ** 2 + (row["point"][1] - lat) ** 2,
        abs((body.start_date - date.fromisoformat(row["date"])).days),
    ))
    return row


def offline():
    return os.getenv("SATQUERY_OFFLINE") == "1"


def resolve_point(client, body):
    if body.place is not None and offline():
        return tuple(backup_for(body, None)["point"]), "Offline mode is enabled."
    try:
        return point_for(client, body), None
    except GeocodeError as error:
        return tuple(backup_for(body, None)["point"]), str(error)
