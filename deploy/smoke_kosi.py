#!/usr/bin/env python3
"""GET-only Kosi deployment check. Usage: python3 deploy/smoke_kosi.py [BASE_URL]."""

import hashlib
import json
import os
import sys
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

METADATA_PATH = '/api/sar/change/kosi-2024'
RASTER_PATH = METADATA_PATH + '/raster'
RASTER_SHA256 = '7d63525d8851c695c9f1e5e72719b80aebd3e6d18581e875bc60ba5d28f6d3a4'
RASTER_SIZE = 13_007_495


class NoRedirects(HTTPRedirectHandler):
    # Check the configured origin itself; redirects must not hide routing errors.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(base_url):
    base_url = base_url.rstrip('/')
    origin = urlsplit(base_url)
    require(origin.scheme in ('http', 'https') and origin.hostname
            and not origin.username and not origin.password
            and not origin.path and not origin.query and not origin.fragment,
            'BASE_URL must be an HTTP(S) origin without credentials, path, query or fragment')
    opener = build_opener(NoRedirects())
    with opener.open(base_url + METADATA_PATH, timeout=60) as response:
        require(response.status == 200, 'Metadata endpoint must return HTTP 200')
        report = json.load(response)
    require(isinstance(report, dict), 'Metadata must be an object')
    expected = {
        'artifact_id': 'kosi-2024-sar-change',
        'scientific_label': 'Measured SAR backscatter change; flood interpretation unvalidated.',
        'raster_download_url': RASTER_PATH,
    }
    for key, value in expected.items():
        require(report.get(key) == value, f'Incorrect {key}')
    require(report.get('flood_interpretation_validated') is False,
            'flood_interpretation_validated must be false')
    digest = hashlib.sha256()
    size = 0
    with opener.open(base_url + RASTER_PATH, timeout=60) as response:
        require(response.status == 200, 'Raster endpoint must return HTTP 200')
        require(response.headers.get_content_type() == 'image/tiff', 'Expected Content-Type image/tiff')
        for chunk in iter(lambda: response.read(65536), b''):
            size += len(chunk)
            require(size <= RASTER_SIZE, 'Raster exceeds expected byte size')
            digest.update(chunk)
    require(size == RASTER_SIZE, f'Incorrect raster byte size: {size}')
    require(digest.hexdigest() == RASTER_SHA256, 'Raster SHA256 mismatch')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        sys.exit('usage: python3 deploy/smoke_kosi.py [BASE_URL]')
    default = ('https://' + os.environ['DOMAIN']) if os.environ.get('DOMAIN') else (
        'http://localhost:' + os.environ.get('HTTP_PORT', '80'))
    try:
        check(sys.argv[1] if len(sys.argv) == 2 else os.environ.get('BASE_URL', default))
    except (ValueError, OSError, URLError) as exc:
        sys.exit(f'FAIL Kosi: {exc}')
    print('PASS Kosi: metadata, scientific caveat, raster MIME type, 13,007,495 bytes and SHA256 verified')
