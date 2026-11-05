"""TODO step 5c: threads for a CPU speech instance. Speech alone and speech while the Iris Xe writes, 10 vs 6 threads,
interleaved twice (the first runs showed 4 and 8 threads slower than 10, but they ran later, on a warmer laptop).
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_resource_probe as R  # noqa: E402


if __name__ == "__main__":
    gpu, _ = R.start("vulkan")
    try:
        R.out(case="gpu chat alone", chat=R.chat(gpu, 300))
        for t in (10, 6, 10, 6):
            cpu, _ = R.start("cpu", threads=t)
            try:
                alone = R.tts(cpu)
                res = {}
                th = threading.Thread(target=lambda: res.update(cpu_tts=R.tts(cpu)))
                th.start()
                time.sleep(0.5)
                res["gpu_chat"] = R.chat(gpu, 300)
                th.join()
                R.out(case=f"cpu speech t={t}", tts_alone=alone, **res)
            finally:
                cpu.close()
        R.out(case="gpu chat alone (end)", chat=R.chat(gpu, 300))
    finally:
        gpu.close()
    print(json.dumps({"case": "done"}), flush=True)
