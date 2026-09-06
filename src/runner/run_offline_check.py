"""
Offline Reloadability & Target Compatibility Check.
Verifies that models can be reloaded and run without internet access (HF_HUB_OFFLINE=1).
Also collects system telemetry (CPU arch, instruction set, OS, GLIBC version).
"""

import os
import sys
import time
import json
import platform
import subprocess
import argparse

from ..models.gliner_adapter import GLiNERAdapter
from ..models.uie_adapter import UIEAdapter


def collect_system_telemetry() -> dict:
    telemetry = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "glibc_version": "unknown",
        "cpu_flags": [],
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        telemetry["glibc_version"] = platform.libc_ver()[1]
    except Exception:
        pass

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    telemetry["cpu_flags"] = line.split(":", 1)[1].strip().split()[:20]
                    break
    except Exception:
        pass

    return telemetry


def main():
    parser = argparse.ArgumentParser(description="Run Offline & Compatibility Check")
    parser.add_argument("--model", type=str, required=True, choices=["gliner", "uie-medium", "uie-mini"])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    telemetry = collect_system_telemetry()

    # Enforce strict offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    test_sample = "萧战是萧家的现任族长，他是萧炎的亲生父亲。"
    status = "OFFLINE_SUCCESS"
    error_msg = None
    t0 = time.time()

    try:
        if args.model == "gliner":
            m_path = args.model_path or "fastino/gliner2.5-multi-v1"
            adapter = GLiNERAdapter(m_path, num_threads=1)
            adapter.load_model()
            res = adapter.predict("offline_test", test_sample, config_id="G5", schema_type="novel")
        else:
            variant = "medium" if "medium" in args.model else "mini"
            m_path = args.model_path or f"Casually/uie-{variant}"
            adapter = UIEAdapter(m_path, num_threads=1)
            adapter.load_model()
            res = adapter.predict("offline_test", test_sample, schema_type="novel")
    except Exception as e:
        status = "OFFLINE_FAILED_NETWORK_DEPENDENCY"
        error_msg = str(e)

    duration = time.time() - t0

    report = {
        "model": args.model,
        "offline_status": status,
        "offline_duration_sec": round(duration, 4),
        "error_message": error_msg,
        "system_telemetry": telemetry,
    }

    out_file = os.path.join(args.output_dir, f"offline_check_{args.model}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Offline check finished: Status={status} | Saved to {out_file}")


if __name__ == "__main__":
    main()
