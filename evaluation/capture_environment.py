#!/usr/bin/env python3
"""Record everything a reader needs to reproduce (or fairly judge) your results.

    python evaluation/capture_environment.py --models qwen2.5-coder:3b,qwen2.5-coder:7b --out results/run1/environment.json

Best-effort: anything it cannot find is stored as null, never an error. Put the file in your
supplementary material. Contains no secrets.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import get_json, post_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or r.stderr).strip()[:500] or None
    except Exception:  # noqa: BLE001
        return None


def ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
        if sys.platform == "darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 1024 ** 3, 1)
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                            ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong), ("f", ctypes.c_ulonglong)]
            ms = MS()
            ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def code_fingerprint() -> dict:
    """git commit if this is a repo, plus a hash of backend/app so a zip without .git is still identifiable."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "backend" / "app").rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return {"git_commit": _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "backend_app_sha256": h.hexdigest()[:16]}


def ollama_info(base: str, models: list[str]) -> dict:
    info = {"base_url": base, "version": None, "models": {}}
    try:
        info["version"] = get_json(f"{base}/api/version", timeout=10).get("version")
    except Exception:  # noqa: BLE001
        pass
    for m in models:
        try:
            d = post_json(f"{base}/api/show", {"model": m}, timeout=20).get("details", {})
            info["models"][m] = {k: d.get(k) for k in ("family", "parameter_size", "quantization_level", "format")}
        except Exception:  # noqa: BLE001
            info["models"][m] = None
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--out", default=str(ROOT / "results" / "environment.json"))
    a = ap.parse_args()
    env = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "processor": platform.processor() or None,
                    "cpu_count": os.cpu_count(), "ram_gb": ram_gb(),
                    "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])},
        "software": {"python": sys.version.split()[0], "docker": _run(["docker", "--version"]),
                     "docker_compose": _run(["docker", "compose", "version"]),
                     "ollama_cli": _run(["ollama", "--version"])},
        "ollama": ollama_info(a.ollama.rstrip("/"), [m for m in a.models.split(",") if m.strip()]),
        "code": code_fingerprint(),
        "env_vars": {k: os.environ.get(k) for k in ("OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE",
                                                    "OLLAMA_TEMPERATURE", "OLLAMA_SEED", "OLLAMA_NUM_CTX")},
        "note": "Also store the .env used for the run (without secrets) and the docker-compose.yml.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(env, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
