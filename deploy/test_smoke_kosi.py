"""Local response fixtures; no network, artifacts, ingestion or inference."""
import hashlib
import io
import json
from email.message import Message
from unittest.mock import patch

import pytest

from deploy import smoke_kosi as smoke


class Response(io.BytesIO):
    def __init__(self, body, status=200, mime='image/tiff'):
        super().__init__(body)
        self.status = status
        self.headers = Message()
        self.headers['Content-Type'] = mime


@pytest.mark.parametrize('failure', [None, 'id', 'label', 'flag', 'url', 'metadata_status',
                                     'raster_status', 'mime', 'short', 'long', 'hash', 'json'])
def test_check(failure):
    raster = b'local synthetic raster fixture'
    report = {'artifact_id': 'kosi-2024-sar-change',
              'scientific_label': 'Measured SAR backscatter change; flood interpretation unvalidated.',
              'flood_interpretation_validated': False, 'raster_download_url': smoke.RASTER_PATH}
    for case, key in [('id', 'artifact_id'), ('label', 'scientific_label'),
                      ('flag', 'flood_interpretation_validated'), ('url', 'raster_download_url')]:
        if failure == case:
            report[key] = 0 if case == 'flag' else 'wrong'
    metadata = Response(b'{' if failure == 'json' else json.dumps(report).encode(),
                        status=503 if failure == 'metadata_status' else 200)
    downloaded = raster[:-1] if failure == 'short' else raster + b'x' if failure == 'long' else raster
    response = Response(downloaded, status=404 if failure == 'raster_status' else 200,
                        mime='text/html' if failure == 'mime' else 'image/tiff')
    with patch.object(smoke, 'RASTER_SIZE', len(raster)), patch.object(
            smoke, 'RASTER_SHA256', '0' * 64 if failure == 'hash' else hashlib.sha256(raster).hexdigest()), patch.object(
            smoke, 'build_opener') as build:
        build.return_value.open.side_effect = [metadata, response]
        if failure:
            with pytest.raises(ValueError):
                smoke.check('http://localhost:8765')
        else:
            smoke.check('http://localhost:8765/')
            assert [call.args[0] for call in build.return_value.open.call_args_list] == [
                'http://localhost:8765' + smoke.METADATA_PATH, 'http://localhost:8765' + smoke.RASTER_PATH]
            assert build.return_value.open.call_count == 2


@pytest.mark.parametrize('origin', ['file:///tmp', 'http://localhost/path', 'http://user:pass@localhost',
                                    'http://localhost?x=1', 'http://localhost#fragment'])
def test_invalid_origin_never_requests(origin):
    with patch.object(smoke, 'build_opener') as build, pytest.raises(ValueError):
        smoke.check(origin)
    build.assert_not_called()
