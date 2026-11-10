# Run as a separate target-environment inventory cell; performs no model load/download.
import os, subprocess, sys
weights = os.environ.get("SATQUERY_QWEN_MODEL_DIR", "/kaggle/input/satquery-models/qwen2.5-vl-3b-instruct")
subprocess.run([sys.executable, "pilot.py", "inventory", "--weights", weights], check=True)
