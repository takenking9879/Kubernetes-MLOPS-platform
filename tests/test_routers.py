"""Unit tests for backend routers and services (no network, no S3, no Airflow).

Covers:
  - OrchestrationSelector routing logic
  - JobBuilder dag_conf generation
  - /api/v2/jobs/launch dry_run (httpx AsyncClient, mocked Airflow/S3)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend is importable
_BACKEND = Path(__file__).resolve().parent.parent / "app" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ── OrchestrationSelector ────────────────────────────────────────────────────

from services.orchestration_selector import (  # noqa: E402
    select_training_orchestration,
    select_serving_orchestration,
    generate_recommendations,
)


class TestOrchestrationSelector:
    def test_tabular_types_route_to_ray_train(self):
        for mt in ("ann", "ssm", "bae", "xgboost", "pytorch", "user_uploaded"):
            rec = select_training_orchestration(mt)
            assert rec.orchestration == "ray_train", f"Expected ray_train for {mt}"
            assert rec.dag_id == "training_pipeline_skypilot"

    def test_llm_routes_to_llm_finetune(self):
        rec = select_training_orchestration("llm")
        assert rec.orchestration == "llm_finetune"
        assert rec.dag_id == "llm_training_pipeline"

    def test_unknown_type_defaults_to_ray_train(self):
        rec = select_training_orchestration("my_custom_model")
        assert rec.orchestration == "ray_train"
        assert len(rec.warnings) == 1

    def test_small_model_single_node_routes_vllm_single(self):
        # 7B model (14GB) on single A40 (48GB) → single node
        rec = select_serving_orchestration(
            model_vram_gb=14.0, num_nodes=1, num_gpus_per_node=1, vram_per_gpu_gb=48.0
        )
        assert rec.orchestration == "vllm_single_node"
        assert rec.dag_id == "vllm_serving_pipeline"

    def test_large_model_multi_node_routes_ray_vllm(self):
        # 140GB model on 2 nodes × 8 × 40GB = 640GB → multinode
        rec = select_serving_orchestration(
            model_vram_gb=140.0, num_nodes=2, num_gpus_per_node=8, vram_per_gpu_gb=40.0
        )
        assert rec.orchestration == "ray_vllm_multinode"
        assert rec.dag_id == "ray_vllm_serving_pipeline"

    def test_model_that_doesnt_fit_adds_warning(self):
        # 80GB model on 1 GPU × 40GB → does not fit → warning
        rec = select_serving_orchestration(
            model_vram_gb=80.0, num_nodes=1, num_gpus_per_node=1, vram_per_gpu_gb=40.0
        )
        assert any("GB" in w for w in rec.warnings)

    def test_generate_recommendations_training(self):
        rec = generate_recommendations(job_type="training", model_type="ssm")
        assert rec.orchestration == "ray_train"

    def test_generate_recommendations_serving(self):
        rec = generate_recommendations(
            job_type="serving",
            model_type="llm",
            model_vram_gb=14.0,
            num_nodes=1,
            num_gpus_per_node=1,
            vram_per_gpu_gb=80.0,
        )
        assert rec.orchestration == "vllm_single_node"


# ── JobBuilder ────────────────────────────────────────────────────────────────

from services.job_builder import JobBuilder, TrainingJobConfig, ServingJobConfig  # noqa: E402


class TestJobBuilder:
    def setup_method(self):
        self.builder = JobBuilder()

    def test_tabular_training_returns_dag_conf(self):
        cfg = TrainingJobConfig(model_type="ann", dataset="network_traffic")
        _, dag_conf = self.builder.build_training_job(cfg)
        assert dag_conf["model_type"] == "ann"
        assert dag_conf["dataset"] == "network_traffic"
        assert dag_conf["use_gpu"] is True
        assert dag_conf["num_nodes"] == 1
        # train_run_id auto-generated if not provided
        assert dag_conf["train_run_id"].startswith("train-network_traffic-")

    def test_tabular_training_explicit_run_id(self):
        cfg = TrainingJobConfig(model_type="bae", train_run_id="train-test-abc123")
        _, dag_conf = self.builder.build_training_job(cfg)
        assert dag_conf["train_run_id"] == "train-test-abc123"

    def test_llm_training_adds_llm_fields(self):
        cfg = TrainingJobConfig(
            model_type="llm",
            llm_model_id="Qwen/Qwen2.5-7B",
            max_steps=200,
            lora_enabled=True,
        )
        _, dag_conf = self.builder.build_training_job(cfg)
        assert dag_conf["llm_model_id"] == "Qwen/Qwen2.5-7B"
        assert dag_conf["max_steps"] == 200
        assert dag_conf["lora_enabled"] is True

    def test_no_sky_yaml_path_in_dag_conf(self):
        """sky_yaml_path must NOT be in dag_conf — the file was on the wrong pod."""
        cfg = TrainingJobConfig(model_type="pytorch")
        _, dag_conf = self.builder.build_training_job(cfg)
        assert "sky_yaml_path" not in dag_conf

    def test_serving_single_node_dag_conf(self):
        cfg = ServingJobConfig(
            llm_model_id="Qwen/Qwen2.5-7B",
            vllm_port=8000,
            tensor_parallel_size=1,
        )
        _, dag_conf = self.builder.build_serving_job(cfg, multinode=False)
        assert dag_conf["llm_model_id"] == "Qwen/Qwen2.5-7B"
        assert dag_conf["vllm_port"] == 8000
        assert dag_conf["tensor_parallel_size"] == 1

    def test_serving_multinode_dag_conf(self):
        cfg = ServingJobConfig(
            llm_model_id="meta-llama/Llama-3.1-70B",
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            num_nodes=2,
        )
        _, dag_conf = self.builder.build_serving_job(cfg, multinode=True)
        assert dag_conf["pipeline_parallel_size"] == 2
        assert dag_conf["tensor_parallel_size"] == 8

    def test_yaml_preview_returns_string(self):
        cfg = TrainingJobConfig(model_type="ann")
        preview = self.builder.yaml_preview(cfg)
        assert isinstance(preview, str)
        assert len(preview) > 10

    def test_hyperparams_included_in_dag_conf(self):
        cfg = TrainingJobConfig(
            model_type="xgboost",
            hyperparams={"max_depth": 8, "eta": 0.1},
        )
        _, dag_conf = self.builder.build_training_job(cfg)
        assert dag_conf["hyperparams"]["max_depth"] == 8


# ── jobs.py dry_run endpoint ──────────────────────────────────────────────────

from fastapi.testclient import TestClient  # noqa: E402
from fastapi import FastAPI  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with only the jobs router mounted.

    We stub out _call_airflow (not needed for dry_run) and GPUCatalogService
    (network call) to keep this test hermetic.
    """
    import routers.jobs as _jobs_module

    # Stub the Airflow helper so we never make real HTTP calls
    _jobs_module._call_airflow = MagicMock(return_value=(200, {"dag_run_id": "test-run"}))

    # Stub GPUCatalogService inside _get_any_of to avoid cloud API calls
    with patch(
        "routers.jobs._get_any_of",
        return_value=[{"cloud": "runpod", "accelerators": "RTX4090:1", "use_spot": True}],
    ):
        app = FastAPI()
        from routers.jobs import router
        app.include_router(router)
        with TestClient(app) as c:
            yield c


class TestLaunchEndpoint:
    def test_dry_run_tabular_returns_recommendation(self, client):
        resp = client.post(
            "/api/v2/jobs/launch",
            json={
                "job_type": "training",
                "model": {"model_type": "ann"},
                "training": {
                    "preprocess_run_id": "",
                    "dataset": "network_traffic",
                    "use_gpu": True,
                },
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dry_run"] is True
        assert body["orchestration"] == "ray_train"
        assert body["recommendation"]["dag_id"] == "training_pipeline_skypilot"
        assert isinstance(body["sky_yaml_preview"], str)
        assert body["job_ids"] == {}

    def test_dry_run_llm_returns_llm_recommendation(self, client):
        resp = client.post(
            "/api/v2/jobs/launch",
            json={
                "job_type": "training",
                "model": {"model_type": "llm", "model_id": "Qwen/Qwen2.5-7B"},
                "training": {"max_steps": 100, "lora_enabled": True},
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["orchestration"] == "llm_finetune"

    def test_dry_run_serving_single_node(self, client):
        resp = client.post(
            "/api/v2/jobs/launch",
            json={
                "job_type": "serving",
                "model": {"model_type": "llm", "vram_gb": 14.0},
                "resource_constraints": {
                    "num_gpus_per_node": 1,
                    "min_vram_gb": 48,
                },
                "serving": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "num_nodes": 1,
                },
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["orchestration"] == "vllm_single_node"

    def test_dry_run_serving_multinode(self, client):
        resp = client.post(
            "/api/v2/jobs/launch",
            json={
                "job_type": "serving",
                "model": {"model_type": "llm", "vram_gb": 140.0},
                "resource_constraints": {
                    "num_gpus_per_node": 8,
                    "min_vram_gb": 40,
                    "num_nodes": 2,
                },
                "serving": {
                    "tensor_parallel_size": 8,
                    "pipeline_parallel_size": 2,
                    "num_nodes": 2,
                },
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["orchestration"] == "ray_vllm_multinode"
