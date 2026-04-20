"""
app/backend/services/job_builder.py

Generates Airflow dag_run.conf dicts and YAML previews from structured job
configuration objects.  The DAG uses its own YAML routing and dynamic GPU
catalog; this builder populates dag_conf fields that the DAG expects.

Usage:
    builder = JobBuilder()
    _, dag_conf = builder.build_training_job(config)
    # → caller enriches dag_conf with train_params_s3_path, then triggers DAG
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml as _yaml


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class TrainingJobConfig:
    """All parameters needed to build a training SkyPilot job."""
    model_type: str                            # "ann" | "ssm" | "bae" | "xgboost" | "pytorch" | "llm" | "user_uploaded"
    train_run_id: str = ""                     # generated if empty
    preprocess_run_id: str = ""
    dataset: str = ""
    processed_table: str = ""
    dataset_s3_path: str = ""                  # for llm / user_uploaded

    # GPU resource constraints (any_of list — output of GPUSelectorService)
    any_of: list[dict] = field(default_factory=list)
    num_nodes: int = 1

    # Hyperparameters / model config
    hyperparams: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)

    # Env-level overrides
    use_gpu: bool = True
    use_spot: bool = True
    use_deepspeed: bool = False
    deepspeed_stage: int = 1
    lora_enabled: bool = True
    lora_rank: int = 8
    max_steps: int = 500
    save_steps: int = 100
    hf_token: str = ""
    llm_model_id: str = "meta-llama/Llama-3.1-8B"

    # Custom model upload
    model_architecture_s3: str = ""            # S3 URI to user-uploaded .py


@dataclass
class TabularServingJobConfig:
    """All parameters needed to build a tabular sky serve job."""
    serve_run_id: str = ""
    registry_model_name: str = ""
    alias: str = "champion"
    serve_port: int = 8000
    num_nodes: int = 1
    min_replicas: int = 1
    max_replicas: int = 3
    target_qps_per_replica: int = 10
    any_of: list[dict] = field(default_factory=list)


@dataclass
class ServingJobConfig:
    """All parameters needed to build a vLLM serving SkyPilot job."""
    serve_run_id: str = ""                     # generated if empty
    llm_model_id: str = ""
    hf_token: str = ""
    llm_adapter_s3: str = ""
    vllm_port: int = 8081
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    num_nodes: int = 1
    replicas: int = 1

    # GPU resource constraints
    any_of: list[dict] = field(default_factory=list)


# ── Builder ───────────────────────────────────────────────────────────────────

_S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")  # injected by Airflow pod env var

# Base YAML templates (paths relative to repo root, resolved at build time)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SKY_DIR = _REPO_ROOT / "k3s/sky"

# Tabular training — provider-specific (RunPod/Vast templates, AWS custom image)
_SKY_GPU_RUNPOD_YAML = _SKY_DIR / "ray-gpu-training-runpod.yaml"
_SKY_GPU_VAST_YAML   = _SKY_DIR / "ray-gpu-training-vast.yaml"
_SKY_MULTINODE_YAML  = _SKY_DIR / "ray-gpu-multinode-aws.yaml"

# LLM training — provider-specific
_SKY_LLM_RUNPOD_YAML = _SKY_DIR / "ray-llm-training-runpod.yaml"
_SKY_LLM_VAST_YAML   = _SKY_DIR / "ray-llm-training-vast.yaml"
_SKY_LLM_AWS_YAML    = _SKY_DIR / "ray-llm-training-aws.yaml"

# vLLM serving — single-node provider-specific, multi-node (AWS)
_SKY_VLLM_RUNPOD_YAML = _SKY_DIR / "vllm-serving-runpod.yaml"
_SKY_VLLM_VAST_YAML   = _SKY_DIR / "vllm-serving-vast.yaml"
_SKY_VLLM_MULTI_YAML  = _SKY_DIR / "ray-vllm-multinode-serving.yaml"

# Tabular serving — SkyPilot sky serve (single-node and multi-node Ray)
_SKY_TABULAR_SERVE_SINGLE_YAML = _SKY_DIR / "tabular-serving-single.yaml"
_SKY_TABULAR_SERVE_MULTI_YAML  = _SKY_DIR / "tabular-serving-multinode.yaml"


def _provider_from_any_of(any_of: list[dict]) -> str:
    """Derive the primary provider name from the first any_of entry."""
    if not any_of:
        return "runpod"
    first = any_of[0] or {}
    cloud = str(first.get("cloud", "")).strip().lower()
    if cloud:
        return cloud

    infra = str(first.get("infra", "")).strip().lower()
    if infra:
        return infra.split("/", 1)[0]

    return "runpod"


class JobBuilder:
    """Builds SkyPilot YAML files and Airflow dag_run.conf dicts."""

    # ── Training ──────────────────────────────────────────────────────────────

    def build_training_job(self, config: TrainingJobConfig) -> tuple[str, dict]:
        """
        Returns (yaml_tmp_path, dag_conf).

        yaml_tmp_path: path of the generated /tmp YAML (for sky.Task.from_yaml())
        dag_conf: dict to pass as dag_run.conf when triggering the Airflow DAG
        """
        if not config.train_run_id:
            config.train_run_id = f"train-{config.dataset or 'launch'}-{uuid.uuid4().hex[:6]}"

        is_llm = config.model_type == "llm"
        is_multinode = config.num_nodes > 1 and not is_llm
        provider = _provider_from_any_of(config.any_of)

        if is_multinode:
            base_yaml_path = _SKY_MULTINODE_YAML
        elif is_llm:
            if provider == "vast":
                base_yaml_path = _SKY_LLM_VAST_YAML
            elif provider == "aws":
                base_yaml_path = _SKY_LLM_AWS_YAML
            else:
                base_yaml_path = _SKY_LLM_RUNPOD_YAML
        else:
            base_yaml_path = _SKY_GPU_VAST_YAML if provider == "vast" else _SKY_GPU_RUNPOD_YAML

        sky_conf = self._load_base_yaml(base_yaml_path)

        # Inject dynamic ordered list if provided
        if config.any_of:
            resources_cfg = sky_conf.setdefault("resources", {})
            resources_cfg["ordered"] = config.any_of
            resources_cfg.pop("any_of", None)
            resources_cfg.pop("infra", None)

        # Inject num_nodes for multi-node
        if is_multinode:
            sky_conf["num_nodes"] = config.num_nodes

        # Build env overrides
        env_patch: dict[str, str] = {
            "USE_GPU":          "true" if config.use_gpu else "false",
            "USE_SPOT":         "true" if config.use_spot else "false",
            "USE_DEEPSPEED":    "true" if config.use_deepspeed else "false",
            "DEEPSPEED_STAGE":  str(config.deepspeed_stage),
            "MLFLOW_TRACKING_URI": _MLFLOW_URI,
            "S3_BUCKET":        _S3_BUCKET,
        }
        if is_llm:
            env_patch.update({
                "LLM_TRAIN_RUN_ID":       config.train_run_id,
                "LLM_MODEL_ID":           config.llm_model_id,
                "HF_TOKEN":               config.hf_token,
                "TRAIN_DATASET_S3":       config.dataset_s3_path,
                "LORA_ENABLED":           "true" if config.lora_enabled else "false",
                "LORA_RANK":              str(config.lora_rank),
                "MAX_STEPS":              str(config.max_steps),
                "SAVE_STEPS":             str(config.save_steps),
            })
        else:
            if config.model_architecture_s3:
                env_patch["MODEL_ARCHITECTURE_S3"] = config.model_architecture_s3

        sky_conf.setdefault("envs", {}).update(env_patch)

        dag_conf: dict[str, Any] = {
            "train_run_id":         config.train_run_id,
            "preprocess_run_id":    config.preprocess_run_id,
            "dataset":              config.dataset,
            "processed_table":      config.processed_table,
            "model_type":           config.model_type,
            "use_gpu":              config.use_gpu,
            "use_deepspeed":        config.use_deepspeed,
            "deepspeed_stage":      config.deepspeed_stage,
            "num_nodes":            config.num_nodes,
        }
        if is_llm:
            dag_conf.update({
                "llm_model_id":     config.llm_model_id,
                "hf_token":         config.hf_token,
                "train_dataset_s3": config.dataset_s3_path,
                "max_steps":        config.max_steps,
                "lora_enabled":     config.lora_enabled,
            })
        if config.hyperparams:
            dag_conf["hyperparams"] = config.hyperparams
        if config.model_config:
            dag_conf["model_config"] = config.model_config

        return "", dag_conf

    # ── Tabular serving (sky serve) ───────────────────────────────────────────

    def build_tabular_serving_job(self, config: TabularServingJobConfig) -> tuple[str, dict]:
        """Returns (yaml_tmp_path, dag_conf) for a tabular sky serve job."""
        if not config.serve_run_id:
            config.serve_run_id = f"serve-tabular-{uuid.uuid4().hex[:8]}"

        base_yaml_path = (
            _SKY_TABULAR_SERVE_MULTI_YAML if config.num_nodes > 1
            else _SKY_TABULAR_SERVE_SINGLE_YAML
        )
        sky_conf = self._load_base_yaml(base_yaml_path)

        # Inject replica policy
        sky_conf.setdefault("service", {}).setdefault("replica_policy", {}).update({
            "min_replicas": config.min_replicas,
            "max_replicas": config.max_replicas,
            "target_qps_per_replica": config.target_qps_per_replica,
        })
        if config.num_nodes > 1:
            sky_conf["num_nodes"] = config.num_nodes
        if config.any_of:
            resources_cfg = sky_conf.setdefault("resources", {})
            resources_cfg["ordered"] = config.any_of
            resources_cfg.pop("any_of", None)

        sky_conf.setdefault("envs", {}).update({
            "SERVE_RUN_ID":         config.serve_run_id,
            "REGISTRY_MODEL_NAME":  config.registry_model_name,
            "MODEL_ALIAS":          config.alias,
            "SERVE_PORT":           str(config.serve_port),
            "MLFLOW_TRACKING_URI":  _MLFLOW_URI,
        })

        dag_conf: dict[str, Any] = {
            "serve_run_id":           config.serve_run_id,
            "registry_model_name":    config.registry_model_name,
            "mlflow_tracking_uri":    _MLFLOW_URI,
            "alias":                  config.alias,
            "serve_port":             config.serve_port,
            "num_nodes":              config.num_nodes,
            "min_replicas":           config.min_replicas,
            "max_replicas":           config.max_replicas,
            "target_qps_per_replica": config.target_qps_per_replica,
        }
        if config.any_of:
            dag_conf["resource_constraints"] = {"ordered": config.any_of}
        return "", dag_conf

    # ── vLLM Serving ──────────────────────────────────────────────────────────

    def build_serving_job(self, config: ServingJobConfig, multinode: bool = False) -> tuple[str, dict]:
        """
        Returns (yaml_tmp_path, dag_conf) for a vLLM serving job.
        """
        if not config.serve_run_id:
            config.serve_run_id = f"serve-{uuid.uuid4().hex[:8]}"

        if multinode:
            base_yaml_path = _SKY_VLLM_MULTI_YAML
        else:
            provider = _provider_from_any_of(config.any_of)
            base_yaml_path = _SKY_VLLM_VAST_YAML if provider == "vast" else _SKY_VLLM_RUNPOD_YAML
        sky_conf = self._load_base_yaml(base_yaml_path)

        if config.any_of:
            resources_cfg = sky_conf.setdefault("resources", {})
            resources_cfg["ordered"] = config.any_of
            resources_cfg.pop("any_of", None)
            resources_cfg.pop("infra", None)

        if multinode and config.num_nodes > 1:
            sky_conf["num_nodes"] = config.num_nodes
        if multinode:
            sky_conf.setdefault("service", {})["replicas"] = config.replicas

        sky_conf.setdefault("envs", {}).update({
            "HF_MODEL_ID":              config.llm_model_id,
            "HF_TOKEN":                 config.hf_token,
            "LLM_ADAPTER_S3":          config.llm_adapter_s3,
            "VLLM_PORT":               str(config.vllm_port),
            "MAX_MODEL_LEN":           str(config.max_model_len),
            "TENSOR_PARALLEL_SIZE":    str(config.tensor_parallel_size),
            "PIPELINE_PARALLEL_SIZE":  str(config.pipeline_parallel_size),
        })

        dag_conf: dict[str, Any] = {
            "serve_run_id":           config.serve_run_id,
            "llm_model_id":           config.llm_model_id,
            "hf_token":               config.hf_token,
            "llm_adapter_s3":         config.llm_adapter_s3,
            "vllm_port":              config.vllm_port,
            "max_model_len":          config.max_model_len,
            "tensor_parallel_size":   config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "replicas":               config.replicas,
        }
        return "", dag_conf

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_base_yaml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path) as fh:
            return _yaml.safe_load(fh) or {}

    def yaml_preview(self, config: TrainingJobConfig | ServingJobConfig) -> str:
        """Return the generated YAML as a string (for UI preview)."""
        is_serving = isinstance(config, ServingJobConfig)
        any_of = config.any_of if hasattr(config, "any_of") else []
        provider = _provider_from_any_of(any_of)

        if is_serving:
            assert isinstance(config, ServingJobConfig)
            if config.num_nodes > 1:
                base = _SKY_VLLM_MULTI_YAML
            elif provider == "vast":
                base = _SKY_VLLM_VAST_YAML
            else:
                base = _SKY_VLLM_RUNPOD_YAML
        elif isinstance(config, TrainingJobConfig) and config.num_nodes > 1:
            base = _SKY_MULTINODE_YAML
        elif isinstance(config, TrainingJobConfig) and config.model_type == "llm":
            if provider == "vast":
                base = _SKY_LLM_VAST_YAML
            elif provider == "aws":
                base = _SKY_LLM_AWS_YAML
            else:
                base = _SKY_LLM_RUNPOD_YAML
        else:
            base = _SKY_GPU_VAST_YAML if provider == "vast" else _SKY_GPU_RUNPOD_YAML

        sky_conf = self._load_base_yaml(base)
        if any_of:
            resources_cfg = sky_conf.setdefault("resources", {})
            resources_cfg["ordered"] = any_of
            resources_cfg.pop("any_of", None)
        return _yaml.dump(sky_conf, default_flow_style=False, allow_unicode=True)
