"""Offline-contract verification for the cached Streamlit golden path."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from demo_gui import golden_assets

APP_PATH = Path(__file__).with_name("app.py")


class OfflineGoldenPathTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()

    def tearDown(self) -> None:
        st.cache_data.clear()

    def test_cached_golden_path_forces_offline_mode_and_never_connects(self) -> None:
        hostile_environment = {
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
        }
        with (
            patch.dict(os.environ, hostile_environment, clear=False),
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("cached golden path attempted network access"),
            ) as connect,
            patch.object(golden_assets, "local_golden_image", return_value=None),
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            next(button for button in app.button if button.label == "Ask").click()
            app.run(timeout=30)

            self.assertFalse(list(app.exception))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            connect.assert_not_called()
            self.assertIn("VERIFIED CACHED RESULT", [item.value for item in app.info])
            self.assertIn("Yes", [item.value for item in app.markdown])


if __name__ == "__main__":
    unittest.main()
