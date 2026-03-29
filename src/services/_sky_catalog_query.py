#!/usr/bin/env python3
"""
src/services/_sky_catalog_query.py

Subprocess helper executed inside the isolated SkyPilot venv (/opt/sky-venv).
Outputs JSON to stdout: { gpu_name: [{cloud, region, price, spot_price,
                                       accelerator_count, device_memory,
                                       cpu_count, memory, instance_type}] }

Includes aws, runpod, vast (COMMON + OTHER GPUs — all=True removed in SkyPilot 0.12.0).
GCP/Azure excluded for now.

Usage (from gpu_catalog.py):
    subprocess.run(["/opt/sky-venv/bin/python", "<this_file>"], ...)

Equivalently:
    sky show-gpus --infra aws -a
    sky show-gpus --infra runpod -a
    sky show-gpus --infra vast -a
"""
import json
import math
import sys

import sky  # type: ignore[import]

VALID_CLOUDS = {"aws", "runpod", "vast"}


def _f(v):
    """Parse float; return None on NaN/None/error."""
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _i(v, default=0):
    """Parse int; return default on error."""
    x = _f(v)
    return int(x) if x is not None else default


out: dict = {}
# sky.list_accelerators() returns all GPU types (COMMON + OTHER) by default.
# The `all` parameter was removed in SkyPilot 0.12.0.
for gpu_name, instances in sky.list_accelerators(
    gpus_only=True, clouds=list(VALID_CLOUDS)
).items():
    rows = []
    for inst in instances:
        cloud = str(inst.cloud or "").lower().split(".")[-1]
        if cloud not in VALID_CLOUDS:
            continue
        rows.append({
            "cloud":             cloud,
            "price":             _f(getattr(inst, "price", 0)) or 0.0,
            "spot_price":        _f(getattr(inst, "spot_price", None)),
            "accelerator_count": _i(getattr(inst, "accelerator_count", 1), 1),
            "device_memory":     _f(getattr(inst, "device_memory", 0)) or 0.0,
            "cpu_count":         _i(getattr(inst, "cpu_count", 0)),
            "memory":            _f(getattr(inst, "memory", 0)) or 0.0,
            "region":            str(getattr(inst, "region", "") or ""),
            "instance_type":     str(getattr(inst, "instance_type", "") or ""),
        })
    if rows:
        out[gpu_name] = rows

json.dump(out, sys.stdout)
