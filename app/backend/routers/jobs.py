"""
app/backend/routers/jobs.py

Unified job launch, status, and cancellation API (Phase 8).

Endpoints:
  POST   /api/v2/jobs/launch           — launch training or serving job
  GET    /api/v2/jobs/{job_id}/status  — poll job status
  GET    /api/v2/jobs/                 — list recent jobs
  DELETE /api/v2/jobs/{job_id}         — cancel / terminate

The launch endpoint calls OrchestrationSelector → JobBuilder → triggers
the appropriate Airflow DAG.  For tabular training the endpoint also
generates params_training.yaml (with lineage when preprocess_run_id is
supplied) and uploads it to S3 so the DAG can find it via
conf["train_params_s3_path"].
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import yaml as _yaml
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Ensure project root on path so services/ and src/ are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.orchestration_selector import (  # noqa: E402
    OrchestratorRecommendation,
    generate_recommendations,
)
from services.job_builder import (  # noqa: E402
    JobBuilder,
    TrainingJobConfig,
    ServingJobConfig,
)

# Reuse Airflow helpers + S3 helpers from runs.py to avoid duplication
from routers.runs import (  # noqa: E402
    _airflow_request as _call_airflow,
    _fetch_preprocess_params,
    _build_lineage,
)

router = APIRouter(prefix="/api/v2/jobs", tags=["jobs"])

# ── Constants ─────────────────────────────────────────────────────────────────

_S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
_AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

_builder = JobBuilder()


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=_AWS_REGION,
    )


# ── Request / Response models ─────────────────────────────────────────────────


class ResourceConstraintsIn(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["runpod"])
    gpu_types: list[str] | None = None
    min_vram_gb: float = 0
    max_price_per_hour: float = 9999
    prefer_spot: bool = True
    require_infiniband: bool = False
    preferred_regions: list[str] = Field(default_factory=list)
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    job_type: str = "tabular"


class ModelConfigIn(BaseModel):
    model_type: str = "pytorch"          # ann | ssm | bae | xgboost | pytorch | llm | user_uploaded
    model_id: str = ""                   # HF model_id for LLM
    architecture_s3: str = ""            # S3 path for user-uploaded .py
    vram_gb: float = 0.0                 # known VRAM requirement (for serving recommendation)


class TrainingConfigIn(BaseModel):
    preprocess_run_id: str = ""
    dataset: str = ""
    processed_table: str = ""
    dataset_s3_path: str = ""            # JSONL for LLM
    use_gpu: bool = True
    use_deepspeed: bool = False
    deepspeed_stage: int = 1
    lora_enabled: bool = True
    lora_rank: int = 8
    max_steps: int = 500
    save_steps: int = 100
    hf_token: str = ""
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    model_cfg: dict[str, Any] = Field(default_factory=dict)
    num_nodes: int = 1


class ServingConfigIn(BaseModel):
    hf_token: str = ""
    llm_adapter_s3: str = ""
    vllm_port: int = 8081
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    num_nodes: int = 1


class LaunchRequest(BaseModel):
    job_type: str                              # "training" | "serving" | "both"
    model: ModelConfigIn = Field(default_factory=ModelConfigIn)
    resource_constraints: ResourceConstraintsIn | None = None
    training: TrainingConfigIn = Field(default_factory=TrainingConfigIn)
    serving: ServingConfigIn = Field(default_factory=ServingConfigIn)
    dry_run: bool = False                      # return YAML preview without triggering


class RecommendationOut(BaseModel):
    orchestration: str
    dag_id: str
    sky_yaml_template: str
    reason: str
    estimated_cost_spot: float | None
    estimated_cost_ondemand: float | None
    warnings: list[str]


class LaunchResponse(BaseModel):
    job_ids: dict[str, str]                    # {\"training\": dag_run_id, \"serving\": dag_run_id}
    orchestration: str
    recommendation: RecommendationOut
    sky_yaml_preview: str                      # generated YAML as string
    dry_run: bool


class JobStatusResponse(BaseModel):
    job_id: str
    dag_run_id: str
    dag_id: str
    state: str
    start_date: str | None
    end_date: str | None


# ── Airflow helpers ───────────────────────────────────────────────────────────


def _trigger_dag(dag_id: str, conf: dict) -> str:
    """Trigger an Airflow DAG run and return dag_run_id."""
    status, data = _call_airflow(
        "POST",
        f"api/v2/dags/{dag_id}/dagRuns",
        body={"logical_date": datetime.now(timezone.utc).isoformat(), "conf": conf},
    )
    if status >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Airflow trigger failed ({status}): {data}",
        )
    return data.get("dag_run_id", "")


def _get_dag_status(dag_id: str, dag_run_id: str) -> dict:
    """Fetch DAG run status from Airflow."""
    status, data = _call_airflow(
        "GET",
        f"api/v2/dags/{dag_id}/dagRuns/{dag_run_id}",
    )
    if status >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Airflow status fetch failed ({status}): {data}",
        )
    return data


# ── Training params helpers ───────────────────────────────────────────────────


def _make_train_run_id(body: LaunchRequest) -> str:
    dataset = body.training.dataset or "launch"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"train-{dataset}-{ts}Z-{rand}"


def _upload_training_params(
    body: LaunchRequest,
    train_run_id: str,
    lineage: dict | None,
) -> str:
    """
    Build a params_training.yaml from the launch request, upload to S3, and
    return the s3:// path.  The DAG reads this via PARAMS_S3_PATH env var.
    """
    model_cfg = body.training.model_cfg or {}
    processed_table = (
        (lineage or {}).get("processed_table")
        or body.training.processed_table
        or model_cfg.get("processed_table", "")
    )

    params: dict[str, Any] = {
        "run_metadata": {
            "run_id": train_run_id,
            "run_type": "training",
            "dataset": body.training.dataset or model_cfg.get("dataset", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "execution": {
            "train_run_id": train_run_id,
            "processed_table": processed_table,
        },
        "kuberay": {
            "model": {
                "framework": body.model.model_type,
                "model_type": body.model.model_type,
                "use_gpu": body.training.use_gpu,
                "target": model_cfg.get("target", ""),
                "num_classes": model_cfg.get("num_classes", 2),
                "task_type": model_cfg.get("task_type", "classification"),
                "input_dim": model_cfg.get("input_dim", 14),
                "dsl_count_dim": True,
                "seed": model_cfg.get("seed", 42),
                "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", ""),
                "mlflow_experiment_name": model_cfg.get(
                    "experiment_name",
                    f"launch_{body.model.model_type}",
                ),
                "mlflow_artifact_location": f"s3://{_S3_BUCKET}/mlflow-artifacts/",
                "mlflow_registry_model_name": model_cfg.get(
                    "registry_model_name",
                    body.model.model_type,
                ),
            }
        },
        "hyperparams": {body.model.model_type: body.training.hyperparams or {}},
        "iceberg_tables": {
            "warehouse": f"s3://{_S3_BUCKET}/warehouse",
            "processed": {"catalog": "iceberg", "namespace": "processed"},
        },
    }

    # Always write a lineage block — serving_configs.py reads lineage.dataset
    # to generate serve_run_id.  Populate from full lineage when available, or
    # from the request fields when no preprocess_run_id was supplied.
    dataset = body.training.dataset or model_cfg.get("dataset", "")
    params["lineage"] = lineage or {
        "preprocess_run_id": "",
        "processed_table": processed_table,
        "dataset": dataset,
    }
    if lineage:
        params["execution"]["preprocess_run_id"] = lineage.get("preprocess_run_id", "")
        params["splits"] = lineage.get("splits", {})

    yaml_str = _yaml.dump(params, default_flow_style=False, allow_unicode=True)
    key = f"runs/training/{train_run_id}/params_training.yaml"
    _s3_client().put_object(
        Bucket=_S3_BUCKET,
        Key=key,
        Body=yaml_str.encode("utf-8"),
        ContentType="application/x-yaml",
    )
    return f"s3://{_S3_BUCKET}/{key}"


# ── Misc helpers ──────────────────────────────────────────────────────────────


def _rec_to_out(rec: OrchestratorRecommendation) -> RecommendationOut:
    return RecommendationOut(
        orchestration=rec.orchestration,
        dag_id=rec.dag_id,
        sky_yaml_template=rec.sky_yaml_template,
        reason=rec.reason,
        estimated_cost_spot=rec.estimated_cost_spot,
        estimated_cost_ondemand=rec.estimated_cost_ondemand,
        warnings=rec.warnings,
    )


def _get_any_of(constraints: ResourceConstraintsIn | None) -> list[dict]:
    """Query GPUSelectorService for the any_of list (used only for YAML preview)."""
    if not constraints:
        return []
    try:
        import dataclasses
        from src.services.gpu_catalog import GPUCatalogService
        from src.services.gpu_selector import GPUSelectorService, ResourceConstraints

        _valid = {f.name for f in dataclasses.fields(ResourceConstraints)}
        merged = {k: v for k, v in constraints.model_dump().items() if k in _valid}
        rc = ResourceConstraints(**merged)
        offers = GPUCatalogService().query_availability(
            providers=rc.providers,
            min_vram_gb=rc.min_vram_gb,
            gpu_types=rc.gpu_types,
        )
        result = GPUSelectorService().select_providers(rc, offers)
        return result.any_of
    except Exception:
        return []


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/launch", response_model=LaunchResponse)
async def launch_job(body: LaunchRequest) -> LaunchResponse:
    """
    Launch a training, serving, or training+serving job.

    - Selects orchestration (ray_train / llm_finetune / vllm_single / ray_vllm_multi)
    - Builds SkyPilot YAML preview from constraints
    - For tabular training: generates params_training.yaml and uploads to S3
    - Triggers the appropriate Airflow DAG(s)
    - Returns YAML preview + DAG run IDs
    """
    any_of = _get_any_of(body.resource_constraints)

    # ── Recommendation ────────────────────────────────────────────────────────
    is_serving = body.job_type in ("serving", "both")
    is_training = body.job_type in ("training", "both")

    vram_per_gpu = (
        body.resource_constraints.min_vram_gb
        if body.resource_constraints and body.resource_constraints.min_vram_gb > 0
        else 16.0
    )
    num_nodes_serving = body.serving.num_nodes if is_serving else 1
    gpus_per_node_serving = (
        body.resource_constraints.num_gpus_per_node
        if body.resource_constraints else 1
    )

    rec_job_type = "serving" if body.job_type == "serving" else "training"
    rec = generate_recommendations(
        job_type=rec_job_type,
        model_type=body.model.model_type,
        model_vram_gb=body.model.vram_gb,
        num_nodes=num_nodes_serving,
        num_gpus_per_node=gpus_per_node_serving,
        vram_per_gpu_gb=vram_per_gpu,
    )

    # ── YAML preview ──────────────────────────────────────────────────────────
    if is_training:
        train_cfg = TrainingJobConfig(
            model_type=body.model.model_type,
            llm_model_id=body.model.model_id,
            any_of=any_of,
            num_nodes=body.training.num_nodes,
            preprocess_run_id=body.training.preprocess_run_id,
            dataset=body.training.dataset,
            processed_table=body.training.processed_table,
            dataset_s3_path=body.training.dataset_s3_path,
            use_gpu=body.training.use_gpu,
            use_deepspeed=body.training.use_deepspeed,
            deepspeed_stage=body.training.deepspeed_stage,
            lora_enabled=body.training.lora_enabled,
            lora_rank=body.training.lora_rank,
            max_steps=body.training.max_steps,
            save_steps=body.training.save_steps,
            hf_token=body.training.hf_token,
            hyperparams=body.training.hyperparams,
            model_config=body.training.model_cfg,
            model_architecture_s3=body.model.architecture_s3,
        )
        yaml_preview = _builder.yaml_preview(train_cfg)
    else:
        serve_cfg = ServingJobConfig(
            llm_model_id=body.model.model_id,
            hf_token=body.serving.hf_token,
            llm_adapter_s3=body.serving.llm_adapter_s3,
            vllm_port=body.serving.vllm_port,
            max_model_len=body.serving.max_model_len,
            tensor_parallel_size=body.serving.tensor_parallel_size,
            pipeline_parallel_size=body.serving.pipeline_parallel_size,
            num_nodes=body.serving.num_nodes,
            any_of=any_of,
        )
        yaml_preview = _builder.yaml_preview(serve_cfg)

    if body.dry_run:
        return LaunchResponse(
            job_ids={},
            orchestration=rec.orchestration,
            recommendation=_rec_to_out(rec),
            sky_yaml_preview=yaml_preview,
            dry_run=True,
        )

    # ── Launch ────────────────────────────────────────────────────────────────
    job_ids: dict[str, str] = {}

    if is_training:
        is_llm = body.model.model_type == "llm"

        if is_llm:
            # LLM DAG uses its own conf — no params_training.yaml needed
            _, dag_conf = _builder.build_training_job(
                TrainingJobConfig(
                    model_type=body.model.model_type,
                    llm_model_id=body.model.model_id,
                    any_of=any_of,
                    num_nodes=body.training.num_nodes,
                    preprocess_run_id=body.training.preprocess_run_id,
                    dataset=body.training.dataset,
                    processed_table=body.training.processed_table,
                    dataset_s3_path=body.training.dataset_s3_path,
                    use_gpu=body.training.use_gpu,
                    use_deepspeed=body.training.use_deepspeed,
                    deepspeed_stage=body.training.deepspeed_stage,
                    lora_enabled=body.training.lora_enabled,
                    lora_rank=body.training.lora_rank,
                    max_steps=body.training.max_steps,
                    save_steps=body.training.save_steps,
                    hf_token=body.training.hf_token,
                    hyperparams=body.training.hyperparams,
                    model_config=body.training.model_cfg,
                    model_architecture_s3=body.model.architecture_s3,
                )
            )
            if body.resource_constraints:
                dag_conf["resource_constraints"] = body.resource_constraints.model_dump()
            train_dag_id = "llm_training_pipeline"
        else:
            # Tabular: generate params_training.yaml → include train_params_s3_path in dag_conf
            train_run_id = _make_train_run_id(body)

            # Build lineage if preprocessing run ID supplied
            lineage: dict | None = None
            if body.training.preprocess_run_id:
                try:
                    s3 = _s3_client()
                    pre_params = _fetch_preprocess_params(
                        body.training.preprocess_run_id, s3, _S3_BUCKET
                    )
                    lineage = _build_lineage(
                        body.training.preprocess_run_id, pre_params, _S3_BUCKET
                    )
                except HTTPException:
                    raise  # 404 if preprocess_run_id doesn't exist — caller should fix

            train_params_s3_path = _upload_training_params(body, train_run_id, lineage)

            _, dag_conf = _builder.build_training_job(
                TrainingJobConfig(
                    model_type=body.model.model_type,
                    train_run_id=train_run_id,
                    llm_model_id=body.model.model_id,
                    any_of=any_of,
                    num_nodes=body.training.num_nodes,
                    preprocess_run_id=body.training.preprocess_run_id,
                    dataset=body.training.dataset,
                    processed_table=body.training.processed_table,
                    dataset_s3_path=body.training.dataset_s3_path,
                    use_gpu=body.training.use_gpu,
                    use_deepspeed=body.training.use_deepspeed,
                    deepspeed_stage=body.training.deepspeed_stage,
                    lora_enabled=body.training.lora_enabled,
                    lora_rank=body.training.lora_rank,
                    max_steps=body.training.max_steps,
                    save_steps=body.training.save_steps,
                    hf_token=body.training.hf_token,
                    hyperparams=body.training.hyperparams,
                    model_config=body.training.model_cfg,
                    model_architecture_s3=body.model.architecture_s3,
                )
            )
            dag_conf["train_params_s3_path"] = train_params_s3_path
            if body.resource_constraints:
                dag_conf["resource_constraints"] = body.resource_constraints.model_dump()
            train_dag_id = "training_pipeline_skypilot"

        job_ids["training"] = _trigger_dag(train_dag_id, dag_conf)

    if is_serving:
        multinode = rec.orchestration == "ray_vllm_multinode"
        s_cfg = ServingJobConfig(
            llm_model_id=body.model.model_id,
            hf_token=body.serving.hf_token,
            llm_adapter_s3=body.serving.llm_adapter_s3,
            vllm_port=body.serving.vllm_port,
            max_model_len=body.serving.max_model_len,
            tensor_parallel_size=body.serving.tensor_parallel_size,
            pipeline_parallel_size=body.serving.pipeline_parallel_size,
            num_nodes=body.serving.num_nodes,
            any_of=any_of,
        )
        _, serve_dag_conf = _builder.build_serving_job(s_cfg, multinode=multinode)
        if body.resource_constraints:
            serve_dag_conf["resource_constraints"] = body.resource_constraints.model_dump()
        serve_dag_id = rec.dag_id
        job_ids["serving"] = _trigger_dag(serve_dag_id, serve_dag_conf)

    return LaunchResponse(
        job_ids=job_ids,
        orchestration=rec.orchestration,
        recommendation=_rec_to_out(rec),
        sky_yaml_preview=yaml_preview,
        dry_run=False,
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    dag_id: str = "training_pipeline_skypilot",
) -> JobStatusResponse:
    """
    Poll status for a job (Airflow DAG run).

    Query param `dag_id` selects which DAG to query (default: training_pipeline_skypilot).
    """
    data = _get_dag_status(dag_id, job_id)
    return JobStatusResponse(
        job_id=job_id,
        dag_run_id=data.get("dag_run_id", job_id),
        dag_id=dag_id,
        state=data.get("state", "unknown"),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
    )


@router.get("/")
async def list_jobs(
    dag_id: str = "training_pipeline_skypilot",
    limit: int = 20,
) -> list[dict]:
    """List recent DAG runs from Airflow."""
    status, data = _call_airflow(
        "GET",
        f"api/v2/dags/{dag_id}/dagRuns?order_by=-execution_date&limit={limit}",
    )
    if status >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Airflow list failed ({status}): {data}",
        )
    dag_runs = data.get("dag_runs", [])
    return [
        {
            "job_id": run.get("dag_run_id"),
            "dag_id": dag_id,
            "state": run.get("state"),
            "start_date": run.get("start_date"),
            "end_date": run.get("end_date"),
            "conf": run.get("conf", {}),
        }
        for run in dag_runs
    ]


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    dag_id: str = "training_pipeline_skypilot",
) -> dict:
    """
    Cancel / terminate a running job.

    Sets the Airflow DAG run state to 'failed'.
    For SkyPilot managed jobs the DAG's finally-block calls sky.jobs.cancel().
    """
    status, data = _call_airflow(
        "PATCH",
        f"api/v2/dags/{dag_id}/dagRuns/{job_id}",
        body={"state": "failed"},
    )
    if status >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Airflow cancel failed ({status}): {data}",
        )
    return {"job_id": job_id, "state": "cancelled"}
