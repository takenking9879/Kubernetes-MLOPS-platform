"""Helpers for translating UI GPU constraints into SkyPilot any_of entries.

These helpers are used by multiple routers to ensure user-selected GPU types
and regions are reflected exactly in the generated SkyPilot constraints.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any


try:
    from src.services.provider_region_mapping import (
        RUNPOD_SKYPILOT_REGION_CODES,
        build_infra_from_provider_region,
        normalized_region_code,
    )
except Exception:
    RUNPOD_SKYPILOT_REGION_CODES = frozenset(
        {"CA", "US", "CZ", "IS", "NL", "SE", "RO", "NO"}
    )

    def build_infra_from_provider_region(provider: str, provider_region_id: str) -> str | None:
        p = str(provider or "").strip().lower()
        r = str(provider_region_id or "").strip()

        if p == "runpod":
            if not r:
                return "runpod"
            raw = r.upper()
            parts = [x for x in raw.split("-") if x]
            if raw in RUNPOD_SKYPILOT_REGION_CODES:
                return f"runpod/{raw}"
            if parts and parts[0] in RUNPOD_SKYPILOT_REGION_CODES:
                return f"runpod/{parts[0]}/{raw}"
            if len(parts) >= 2 and parts[1] in RUNPOD_SKYPILOT_REGION_CODES:
                return f"runpod/{parts[1]}"
            return None

        if p == "aws":
            return f"aws/{r.lower()}" if r else "aws"

        if p == "vast":
            return "vast"

        return p if not r else f"{p}/{r.lower()}"

    def normalized_region_code(provider: str, provider_region_id: str) -> str:
        p = str(provider or "").strip().lower()
        r = str(provider_region_id or "").strip()
        if p == "runpod":
            raw = r.upper()
            if raw in RUNPOD_SKYPILOT_REGION_CODES:
                return raw
            parts = [x for x in raw.split("-") if x]
            if parts and parts[0] in RUNPOD_SKYPILOT_REGION_CODES:
                return parts[0]
            if len(parts) >= 2 and parts[1] in RUNPOD_SKYPILOT_REGION_CODES:
                return parts[1]
            return ""
        if p == "aws":
            return r.lower()
        return ""


_RUNPOD_REGION_CODES: frozenset[str] = frozenset(
    RUNPOD_SKYPILOT_REGION_CODES
)


_GPU_TYPE_ALIASES: dict[str, str] = {
    "rtx2000ada": "RTX2000-Ada",
    "nvidiartx2000adageneration": "RTX2000-Ada",
    "rtx4000ada": "RTX4000-Ada",
    "nvidiartx4000adageneration": "RTX4000-Ada",
    "rtx6000ada": "RTX6000-Ada",
    "nvidiartx6000adageneration": "RTX6000-Ada",
    "rtx4090": "RTX4090",
    "nvidiageforcertx4090": "RTX4090",
    "rtx3090": "RTX3090",
    "rtxa4000": "RTXA4000",
    "rtxa4500": "RTXA4500",
    "rtxa5000": "RTXA5000",
    "rtxa6000": "RTXA6000",
    "a100": "A100",
    "a10080gb": "A100-80GB",
    "a10080gbsxm": "A100-80GB-SXM",
    "a40": "A40",
    "h100": "H100",
    "h100nvl": "H100-NVL",
    "h100sxm": "H100-SXM",
    "h200": "H200",
    "h200sxm": "H200-SXM",
    "l4": "L4",
    "l40": "L40",
    "l40s": "L40S",
    "v100": "V100",
    "v10032gb": "V100-32GB",
}


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_provider(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if raw in ("localgpu", "local_gpu"):
        return "localgpu"
    return raw


def _detect_k3s_gpu_accelerator(num_gpus: int = 1) -> str:
    """Return 'rtx4070:1' style string from K3S node labels, or '' on failure."""
    import json as _json
    import subprocess as _sp
    try:
        # Prefer the node label (set by up-k3s-gpu.sh / deploy.sh) — most accurate
        r = _sp.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for node in _json.loads(r.stdout).get("items", []):
                labels = node.get("metadata", {}).get("labels", {})
                cap = node.get("status", {}).get("capacity", {})
                if int(cap.get("nvidia.com/gpu", 0) or 0) == 0:
                    continue
                acc = labels.get("skypilot.co/accelerator", "")
                if acc:
                    return f"{acc}:{num_gpus}"
        # Fallback: nvidia-smi on host
        smi = _sp.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            from src.services.gpu_catalog import _nvidia_name_to_skypilot
            name = _nvidia_name_to_skypilot(smi.stdout.strip().splitlines()[0])
            return f"{name}:{num_gpus}" if name else ""
        return ""
    except Exception:
        return ""


def _normalize_region(provider: str, region: str) -> str:
    p = _normalize_provider(provider)
    r = str(region or "").strip()
    if not r:
        return ""
    return r.upper() if p == "runpod" else r.lower()


def normalize_gpu_type(gpu_type: str) -> str:
    """Normalize UI labels into SkyPilot accelerator names when possible."""
    raw = str(gpu_type or "").strip()
    if not raw:
        return ""

    key = _norm_key(raw)
    alias = _GPU_TYPE_ALIASES.get(key)
    if alias:
        return alias

    # Common compact form: RTX2000Ada -> RTX2000-Ada
    m = re.fullmatch(r"rtx(\d{4})ada", key)
    if m:
        return f"RTX{m.group(1)}-Ada"

    # Keep user intent but remove spaces for SkyPilot accelerator syntax.
    return re.sub(r"\s+", "", raw)


def normalize_accelerator(accelerator: str, default_num_gpus: int = 1) -> str:
    """Normalize an accelerator spec into <GPUType>:<count>."""
    raw = str(accelerator or "").strip()
    if not raw:
        return ""

    if ":" in raw:
        gpu_part, count_part = raw.split(":", 1)
        gpu = normalize_gpu_type(gpu_part)
        count = count_part.strip()
        if count.isdigit() and int(count) > 0:
            return f"{gpu}:{int(count)}"
        return f"{gpu}:{max(1, int(default_num_gpus or 1))}"

    gpu = normalize_gpu_type(raw)
    return f"{gpu}:{max(1, int(default_num_gpus or 1))}"


def _infra_from_provider_region(provider: str, region: str) -> str | None:
    p = _normalize_provider(provider)
    raw = str(region or "").strip()
    mapped = build_infra_from_provider_region(p, raw)
    if mapped is None and raw:
        return None
    return mapped or p


def _dedupe_any_of(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    for e in entries:
        infra = str(e.get("infra", "")).strip()
        acc = str(e.get("accelerators", "")).strip()
        use_spot = bool(e.get("use_spot", False))
        key = (infra, acc, use_spot)
        if not infra or not acc or key in seen:
            continue
        seen.add(key)
        out.append({"infra": infra, "accelerators": acc, "use_spot": use_spot})
    return out


def sanitize_gpu_fallbacks(
    entries: list[dict[str, Any]] | None,
    default_num_gpus: int = 1,
) -> list[dict[str, Any]]:
    """Normalize and validate a user-provided gpu_fallbacks list."""
    if not entries:
        return []

    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        infra_raw = str((entry or {}).get("infra", "")).strip()
        if not infra_raw:
            continue

        parts = infra_raw.split("/")
        parts[0] = _normalize_provider(parts[0])
        infra = "/".join(parts)

        acc = normalize_accelerator(
            str((entry or {}).get("accelerators", "")).strip(),
            default_num_gpus=default_num_gpus,
        )
        if not acc:
            continue

        cleaned.append(
            {
                "infra": infra,
                "accelerators": acc,
                "use_spot": bool((entry or {}).get("use_spot", False)),
            }
        )

    return _dedupe_any_of(cleaned)


def build_strict_any_of(
    providers: list[str] | None,
    gpu_types: list[str] | None,
    preferred_regions: list[str] | None,
    num_gpus_per_node: int = 1,
    prefer_spot: bool = True,
) -> list[dict[str, Any]]:
    """Build a deterministic any_of list from explicit user GPU/region choices."""
    provider_list = [
        _normalize_provider(p) for p in (providers or ["runpod"]) if str(p).strip()
    ]
    gpu_list = [
        normalize_gpu_type(g)
        for g in (gpu_types or [])
        if str(g or "").strip()
    ]

    if not provider_list or not gpu_list:
        return []

    region_raw = [str(r or "").strip() for r in (preferred_regions or []) if str(r).strip()]
    n_gpus = max(1, int(num_gpus_per_node or 1))

    entries: list[dict[str, Any]] = []
    for provider in provider_list:
        if region_raw:
            infra_list: list[str] = []
            for region in region_raw:
                infra = _infra_from_provider_region(provider, region)
                if infra:
                    infra_list.append(infra)
        else:
            infra_list = [_infra_from_provider_region(provider, "") or provider]

        for infra in infra_list:
            for gpu in gpu_list:
                acc = normalize_accelerator(gpu, default_num_gpus=n_gpus)
                if prefer_spot:
                    entries.append(
                        {"infra": infra, "accelerators": acc, "use_spot": True}
                    )
                entries.append(
                    {"infra": infra, "accelerators": acc, "use_spot": False}
                )

    return _dedupe_any_of(entries)


def has_explicit_resource_selection(constraints: dict[str, Any]) -> bool:
    """True when user explicitly selected GPU and/or region constraints."""
    gpu_fallbacks = constraints.get("gpu_fallbacks") or []
    gpu_types = constraints.get("gpu_types") or []
    preferred_regions = constraints.get("preferred_regions") or []
    return bool(gpu_fallbacks or gpu_types or preferred_regions)


def _entry_region(infra: str) -> tuple[str, str]:
    parts = str(infra or "").split("/")
    provider = _normalize_provider(parts[0] if parts else "")
    if provider == "runpod" and len(parts) >= 2:
        return provider, parts[1].upper()
    if provider in {"aws", "gcp", "azure"} and len(parts) >= 2:
        return provider, parts[1].lower()
    return provider, ""


def _filter_any_of_by_regions(
    any_of: list[dict[str, Any]],
    preferred_regions: list[str] | None,
) -> list[dict[str, Any]]:
    if not preferred_regions:
        return any_of

    runpod_regions = {
        normalized_region_code("runpod", str(r).strip())
        for r in preferred_regions
        if str(r).strip()
    }
    runpod_regions.discard("")

    cloud_regions = {
        normalized_region_code("aws", str(r).strip())
        for r in preferred_regions
        if str(r).strip()
    }
    cloud_regions.discard("")

    filtered: list[dict[str, Any]] = []
    for entry in any_of:
        provider, region = _entry_region(str(entry.get("infra", "")))
        if provider == "runpod" and runpod_regions:
            if region in runpod_regions:
                filtered.append(entry)
            continue
        if provider in {"aws", "gcp", "azure"} and cloud_regions:
            if region in cloud_regions:
                filtered.append(entry)
            continue
        filtered.append(entry)

    return filtered


def _build_catalog_any_of(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort catalog-backed any_of construction for auto mode."""
    try:
        from src.services.gpu_catalog import GPUCatalogService
        from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

        valid_fields = {f.name for f in dataclasses.fields(ResourceConstraints)}
        merged = {k: v for k, v in constraints.items() if k in valid_fields}

        # Normalize providers/regions before passing to selector.
        merged["providers"] = [
            _normalize_provider(p) for p in (merged.get("providers") or ["runpod"])
        ]
        merged["preferred_regions"] = [
            str(r).strip() for r in (merged.get("preferred_regions") or []) if str(r).strip()
        ]

        rc = ResourceConstraints(**merged)
        offers = GPUCatalogService().query_availability(
            providers=rc.providers,
            min_vram_gb=rc.min_vram_gb,
            gpu_types=rc.gpu_types,
        )
        result = GPUSelectorService().select_providers(rc, offers)

        n_gpus = max(1, int(getattr(rc, "num_gpus_per_node", 1) or 1))
        any_of = sanitize_gpu_fallbacks(result.any_of, default_num_gpus=n_gpus)
        return _filter_any_of_by_regions(any_of, rc.preferred_regions)
    except Exception:
        return []


def build_any_of_from_constraints(constraints: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Translate generic UI constraints into a SkyPilot any_of list.

    Priority:
      1) explicit gpu_fallbacks (manual mode)
      2) explicit gpu_types (+ optional regions) -> strict deterministic any_of
      3) catalog-backed selector (auto mode, no explicit GPU type)
    """
    if not constraints:
        return []

    _providers = [_normalize_provider(p) for p in (constraints.get("providers") or []) if str(p).strip()]

    # localGPU → K3S in-cluster path: detect GPU from node labels and generate any_of
    if "localgpu" in _providers or "k8s" in _providers:
        n_gpus = max(1, int(constraints.get("num_gpus_per_node") or 1))
        gpu_acc = _detect_k3s_gpu_accelerator(n_gpus)
        if not gpu_acc:
            return []  # cluster unreachable or no labeled GPU node — fail fast
        return [{"infra": "k8s/in-cluster", "accelerators": gpu_acc, "use_spot": False}]

    # Exclusivity guard: localGPU cannot mix with cloud providers
    _local_ids = {"localgpu", "k8s"}
    if any(p in _local_ids for p in _providers) and any(p not in _local_ids for p in _providers):
        raise ValueError("localGPU provider cannot be combined with cloud providers")

    n_gpus = max(1, int(constraints.get("num_gpus_per_node") or 1))
    manual = sanitize_gpu_fallbacks(constraints.get("gpu_fallbacks"), default_num_gpus=n_gpus)
    if manual:
        return manual

    gpu_types = constraints.get("gpu_types") or []
    if gpu_types:
        return build_strict_any_of(
            providers=constraints.get("providers") or ["runpod"],
            gpu_types=gpu_types,
            preferred_regions=constraints.get("preferred_regions") or [],
            num_gpus_per_node=n_gpus,
            prefer_spot=bool(constraints.get("prefer_spot", True)),
        )

    return _build_catalog_any_of(constraints)
