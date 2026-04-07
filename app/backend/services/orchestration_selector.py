"""
app/backend/services/orchestration_selector.py

Selects the appropriate training / serving orchestration backend given a
model type and resource constraints.

Training dispatch:
  ann | ssm | bae | xgboost | user_uploaded  →  ray_train
  llm                                         →  llm_finetune

Serving dispatch:
  model_vram_gb <= num_gpus_per_node * vram_per_gpu_gb AND num_nodes == 1
    → vllm_single_node   (simpler, cheaper)
  else
    → ray_vllm_multinode (Kimi-K2 pattern: Ray head + vLLM tensor+pipeline parallel)
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorRecommendation:
    orchestration: str          # "vllm_single_node" | "ray_vllm_multinode" | "ray_train" | "llm_finetune"
    dag_id: str                 # Airflow DAG to trigger
    sky_yaml_template: str      # Which base SkyPilot YAML to use
    reason: str                 # Human-readable explanation
    estimated_cost_spot: float | None = None
    estimated_cost_ondemand: float | None = None
    warnings: list[str] = field(default_factory=list)


# ── Airflow DAG IDs ───────────────────────────────────────────────────────────

_TRAINING_DAG = "training_pipeline_skypilot"
_LLM_TRAINING_DAG = "llm_training_pipeline"
_VLLM_SERVING_DAG = "vllm_serving_pipeline"
_RAY_VLLM_SERVING_DAG = "ray_vllm_serving_pipeline"
_TABULAR_SERVING_DAG = "tabular_serving_skypilot_pipeline"

# ── SkyPilot YAML templates (display labels — actual file chosen by provider) ─

_SKY_YAML_GPU = "k3s/sky/ray-gpu-training-{runpod|vast}.yaml"
_SKY_YAML_LLM = "k3s/sky/ray-llm-training-{runpod|vast|aws}.yaml"
_SKY_YAML_VLLM_SINGLE = "k3s/sky/vllm-serving-{runpod|vast}.yaml"
_SKY_YAML_VLLM_MULTI = "k3s/sky/ray-vllm-multinode-serving.yaml"
_SKY_YAML_TABULAR_SINGLE = "k3s/sky/tabular-serving-single.yaml"
_SKY_YAML_TABULAR_MULTI = "k3s/sky/tabular-serving-multinode.yaml"


# ── Training orchestration ────────────────────────────────────────────────────

_LLM_MODEL_TYPES = frozenset({"llm"})
_RAY_TRAIN_TYPES = frozenset({"ann", "ssm", "bae", "xgboost", "pytorch", "user_uploaded"})


def select_training_orchestration(model_type: str) -> OrchestratorRecommendation:
    """Return the recommended orchestration for a training job."""
    mt = model_type.lower()
    if mt in _LLM_MODEL_TYPES:
        return OrchestratorRecommendation(
            orchestration="llm_finetune",
            dag_id=_LLM_TRAINING_DAG,
            sky_yaml_template=_SKY_YAML_LLM,
            reason=(
                "LLM fine-tuning uses HuggingFace TRL + DeepSpeed ZeRO directly "
                "(more battle-tested than Ray Train for LLMs). Checkpoints sync to S3 "
                "every SAVE_STEPS for spot recovery."
            ),
        )
    if mt in _RAY_TRAIN_TYPES:
        return OrchestratorRecommendation(
            orchestration="ray_train",
            dag_id=_TRAINING_DAG,
            sky_yaml_template=_SKY_YAML_GPU,
            reason=(
                "Tabular / PyTorch models use Ray Train with ScalingConfig for "
                "distributed training. SkyPilot managed jobs handle spot preemption."
            ),
        )
    # Unknown type — fall back to ray_train
    return OrchestratorRecommendation(
        orchestration="ray_train",
        dag_id=_TRAINING_DAG,
        sky_yaml_template=_SKY_YAML_GPU,
        reason=f"Unknown model type '{model_type}' — defaulting to Ray Train.",
        warnings=[f"Model type '{model_type}' is not explicitly supported."],
    )


# ── Serving orchestration ─────────────────────────────────────────────────────

def select_serving_orchestration(
    model_vram_gb: float,
    num_nodes: int,
    num_gpus_per_node: int,
    vram_per_gpu_gb: float,
) -> OrchestratorRecommendation:
    """
    Return the recommended orchestration for a serving job.

    Single-node if model fits; multi-node (Kimi-K2 Ray+vLLM) otherwise.
    Serving is always on-demand (spot preemption is unacceptable for endpoints).
    """
    total_vram = num_nodes * num_gpus_per_node * vram_per_gpu_gb
    fits_single = model_vram_gb <= num_gpus_per_node * vram_per_gpu_gb and num_nodes == 1
    warnings: list[str] = []

    if model_vram_gb > total_vram:
        warnings.append(
            f"Model requires {model_vram_gb:.0f} GB VRAM but selected config provides "
            f"only {total_vram:.0f} GB. Add more nodes or larger GPUs."
        )

    if fits_single:
        return OrchestratorRecommendation(
            orchestration="vllm_single_node",
            dag_id=_VLLM_SERVING_DAG,
            sky_yaml_template=_SKY_YAML_VLLM_SINGLE,
            reason=(
                f"Model fits on a single node ({model_vram_gb:.0f} GB ≤ "
                f"{num_gpus_per_node}×{vram_per_gpu_gb:.0f} GB). "
                "vLLM alone is simpler and cheaper than Ray+vLLM."
            ),
            warnings=warnings,
        )

    # Multi-node: Kimi-K2 pattern
    if num_nodes > 1 and num_gpus_per_node > 1:
        warnings.append(
            "Multi-node vLLM requires InfiniBand/EFA for good performance. "
            "Select AWS p4d.24xlarge or RunPod NVLink for best results."
        )
    return OrchestratorRecommendation(
        orchestration="ray_vllm_multinode",
        dag_id=_RAY_VLLM_SERVING_DAG,
        sky_yaml_template=_SKY_YAML_VLLM_MULTI,
        reason=(
            f"Model requires {model_vram_gb:.0f} GB VRAM across "
            f"{num_nodes}×{num_gpus_per_node} GPUs. "
            "Ray+vLLM (Kimi-K2 pattern) enables tensor+pipeline parallelism "
            "across nodes."
        ),
        warnings=warnings,
    )


# ── Tabular serving orchestration ────────────────────────────────────────────

def select_tabular_serving_orchestration(num_nodes: int = 1) -> OrchestratorRecommendation:
    """Return the recommended orchestration for SkyPilot tabular serving.

    Single-node: sky serve + Ray Serve, one node per replica.
    Multi-node: sky serve + Ray cluster (head + workers) per replica (Kimi K2 adapted).
    """
    if num_nodes > 1:
        return OrchestratorRecommendation(
            orchestration="tabular_serving_multinode",
            dag_id=_TABULAR_SERVING_DAG,
            sky_yaml_template=_SKY_YAML_TABULAR_MULTI,
            reason=(
                f"Multi-node ({num_nodes} nodes per replica): Ray cluster via SkyPilot sky serve. "
                "Head node loads model + serves via Ray Serve; workers join Ray cluster."
            ),
        )
    return OrchestratorRecommendation(
        orchestration="tabular_serving_single",
        dag_id=_TABULAR_SERVING_DAG,
        sky_yaml_template=_SKY_YAML_TABULAR_SINGLE,
        reason=(
            "Single-node SkyPilot sky serve with Ray Serve. "
            "SkyServe manages replica lifecycle, health checks, and auto-scaling."
        ),
    )


# ── Combined recommendation ───────────────────────────────────────────────────

def generate_recommendations(
    job_type: str,
    model_type: str,
    model_vram_gb: float = 0.0,
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    vram_per_gpu_gb: float = 16.0,
    estimated_cost_spot: float | None = None,
    estimated_cost_ondemand: float | None = None,
) -> OrchestratorRecommendation:
    """
    Top-level entry point.  Returns one recommendation for the given job.

    job_type : "training" | "serving"
    """
    if job_type == "serving":
        rec = select_serving_orchestration(
            model_vram_gb=model_vram_gb,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            vram_per_gpu_gb=vram_per_gpu_gb,
        )
    else:
        rec = select_training_orchestration(model_type)

    rec.estimated_cost_spot = estimated_cost_spot
    rec.estimated_cost_ondemand = estimated_cost_ondemand
    return rec
