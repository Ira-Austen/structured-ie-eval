"""
Background Resource & Cgroup Monitor for Controlled Evaluation.
Polls every 0.5s:
- Host memory (MemAvailable, MemTotal, SwapTotal, SwapFree)
- Process tree RSS / VMS / CPU %
- Cgroup v2 (memory.current, memory.peak, memory.swap.current, memory.events, PSI)
"""

import os
import sys
import time
import json
import threading
import psutil
from typing import Dict, Any, List, Optional


class ResourceMonitor:
    def __init__(self, output_jsonl: Optional[str] = None, interval: float = 0.5):
        self.output_jsonl = output_jsonl
        self.interval = interval
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.samples: List[Dict[str, Any]] = []
        self.peak_rss_mib = 0.0
        self.peak_cgroup_mib = 0.0
        self.peak_swap_mib = 0.0

        self.cgroup_base = self._detect_cgroup_v2_path()

    def _detect_cgroup_v2_path(self) -> Optional[str]:
        try:
            with open(f"/proc/{self.pid}/cgroup", "r") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3 and parts[0] == "0":
                        cgroup_rel = parts[2].lstrip("/")
                        full_path = os.path.join("/sys/fs/cgroup", cgroup_rel)
                        if os.path.exists(full_path):
                            return full_path
                        if os.path.exists("/sys/fs/cgroup"):
                            return "/sys/fs/cgroup"
        except Exception:
            pass
        return None

    def _read_cgroup_file(self, filename: str) -> Optional[str]:
        if not self.cgroup_base:
            return None
        p = os.path.join(self.cgroup_base, filename)
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    return f.read().strip()
            except Exception:
                return None
        return None

    def _poll_once(self) -> Dict[str, Any]:
        now = time.time()
        host_mem = psutil.virtual_memory()
        host_swap = psutil.swap_memory()

        # Process tree memory
        proc_rss = 0
        try:
            proc_rss = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    proc_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

        proc_rss_mib = proc_rss / (1024.0 * 1024.0)
        self.peak_rss_mib = max(self.peak_rss_mib, proc_rss_mib)

        # Cgroup v2 readings
        cg_current_str = self._read_cgroup_file("memory.current")
        cg_peak_str = self._read_cgroup_file("memory.peak")
        cg_swap_str = self._read_cgroup_file("memory.swap.current")
        psi_str = self._read_cgroup_file("memory.pressure")

        cg_current_mib = float(cg_current_str) / (1024.0 * 1024.0) if cg_current_str and cg_current_str.isdigit() else None
        cg_peak_mib = float(cg_peak_str) / (1024.0 * 1024.0) if cg_peak_str and cg_peak_str.isdigit() else None
        cg_swap_mib = float(cg_swap_str) / (1024.0 * 1024.0) if cg_swap_str and cg_swap_str.isdigit() else None

        if cg_peak_mib:
            self.peak_cgroup_mib = max(self.peak_cgroup_mib, cg_peak_mib)
        elif cg_current_mib:
            self.peak_cgroup_mib = max(self.peak_cgroup_mib, cg_current_mib)

        if cg_swap_mib:
            self.peak_swap_mib = max(self.peak_swap_mib, cg_swap_mib)

        sample = {
            "timestamp": round(now, 3),
            "proc_rss_mib": round(proc_rss_mib, 2),
            "host_avail_mib": round(host_mem.available / (1024.0 * 1024.0), 2),
            "host_swap_used_mib": round(host_swap.used / (1024.0 * 1024.0), 2),
            "cgroup_current_mib": round(cg_current_mib, 2) if cg_current_mib is not None else "N/A",
            "cgroup_peak_mib": round(cg_peak_mib, 2) if cg_peak_mib is not None else "N/A",
            "cgroup_swap_mib": round(cg_swap_mib, 2) if cg_swap_mib is not None else "N/A",
            "psi_summary": psi_str.replace("\n", " | ") if psi_str else "N/A"
        }

        return sample

    def _loop(self):
        f = None
        if self.output_jsonl:
            try:
                os.makedirs(os.path.dirname(self.output_jsonl), exist_ok=True)
                f = open(self.output_jsonl, "a", encoding="utf-8")
            except Exception:
                pass

        while self.running:
            try:
                s = self._poll_once()
                self.samples.append(s)
                if f:
                    f.write(json.dumps(s) + "\n")
                    f.flush()
            except Exception:
                pass
            time.sleep(self.interval)

        if f:
            try:
                f.close()
            except Exception:
                pass

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Final reading
        final_sample = self._poll_once()
        return {
            "peak_rss_mib": round(self.peak_rss_mib, 2),
            "peak_cgroup_mib": round(self.peak_cgroup_mib, 2),
            "peak_swap_mib": round(self.peak_swap_mib, 2),
            "total_samples": len(self.samples),
            "last_sample": final_sample
        }
