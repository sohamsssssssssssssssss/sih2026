"""Resume safety for the RSVQA-LR acquisition path.

A partially downloaded `.part` is real progress over a flaky link. The
urlopen path writes with mode "wb", which truncates, so re-entering it on a
retry would silently restart a 95 MB archive from zero — the failure that
made this download never finish on a mobile hotspot. `_download` must route
an existing `.part` to the resuming curl path instead.
"""

import hashlib
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eval.suites import rsvqa

PAYLOAD = b"resumed-archive-bytes"
RESUMED_FROM = b"partial-"


class ResumePreservationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.name = "Images_LR.zip"
        self.entry = {self.name: (len(PAYLOAD), hashlib.md5(PAYLOAD).hexdigest())}
        self.addCleanup(self._tmp.cleanup)

    def _destination(self) -> Path:
        return self.root / self.name

    def test_existing_part_resumes_via_curl_and_is_not_truncated(self) -> None:
        partial = self._destination().with_suffix(".zip.part")
        partial.write_bytes(RESUMED_FROM)
        seen: dict[str, int] = {}

        def fake_curl(url: str, target: Path) -> None:
            # A real resume appends; assert the prior bytes survived to here.
            seen["size_at_entry"] = target.stat().st_size
            target.write_bytes(PAYLOAD)

        with patch.dict(rsvqa.FILES, self.entry, clear=True), \
             patch.object(rsvqa, "_curl_resume", side_effect=fake_curl) as curl, \
             patch.object(urllib.request, "urlopen") as urlopen:
            rsvqa._download(self.name, self._destination())

        curl.assert_called_once()
        urlopen.assert_not_called()
        self.assertEqual(seen["size_at_entry"], len(RESUMED_FROM))
        self.assertEqual(self._destination().read_bytes(), PAYLOAD)

    def test_absent_part_still_uses_the_direct_path(self) -> None:
        class FakeResponse:
            def read(self, *_: object) -> bytes:
                return b""

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

        def fake_copy(_response: object, output: object, length: int = 0) -> None:
            output.write(PAYLOAD)

        with patch.dict(rsvqa.FILES, self.entry, clear=True), \
             patch.object(rsvqa, "_curl_resume") as curl, \
             patch.object(urllib.request, "urlopen", return_value=FakeResponse()), \
             patch.object(rsvqa.shutil, "copyfileobj", side_effect=fake_copy):
            rsvqa._download(self.name, self._destination())

        curl.assert_not_called()
        self.assertEqual(self._destination().read_bytes(), PAYLOAD)


if __name__ == "__main__":
    unittest.main()
