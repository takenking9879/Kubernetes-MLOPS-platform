"""
k3s/kuberay/llm_main.py

LLM fine-tuning entrypoint for SkyPilot-managed jobs.

Stack:
  - HuggingFace Transformers (model + tokenizer)
  - TRL SFTTrainer (supervised fine-tuning)
  - PEFT LoRA (opt-in, default ON)
  - DeepSpeed ZeRO stage 3 (opt-in, default ON for LLMs)
  - boto3 (S3 checkpoint sync + dataset download)
  - MLflow (metrics, tags, adapter path)

Spot recovery:
  - SkyPilot managed jobs (sky.jobs.launch) handle node-level preemption.
  - S3CheckpointCallback syncs each local checkpoint to S3 on every save.
  - On restart, _find_latest_s3_checkpoint() downloads the latest checkpoint
    from S3 and passes it to `resume_from_checkpoint`.
  - Result: restarts from last saved step, not from epoch 0.

Environment variables (set by ray-llm-training.yaml envs):
  LLM_TRAIN_RUN_ID   — unique run ID (e.g. llm-qwen-20260321T120000Z-a1b2c3)
  LLM_MODEL_ID       — HuggingFace model ID (e.g. Qwen/Qwen2.5-7B)
  HF_TOKEN           — HuggingFace token (optional, for gated models)
  TRAIN_DATASET_S3   — s3://bucket/path/train.jsonl
  MAX_SEQ_LEN        — max sequence length (default 2048)
  MAX_STEPS          — training steps (default 500)
  SAVE_STEPS         — checkpoint frequency (default 100)
  PER_DEVICE_BATCH_SIZE — batch size per GPU (default 1)
  GRADIENT_ACCUM_STEPS  — gradient accumulation (default 4)
  LEARNING_RATE      — learning rate (default 2e-4)
  LORA_ENABLED       — "true" | "false" (default "true")
  LORA_RANK          — LoRA rank (default 8)
  USE_DEEPSPEED      — "true" | "false" (default "true")
  DEEPSPEED_STAGE    — ZeRO stage 1|2|3 (default 3)
  USE_SPOT           — "true" | "false" for cost tag (default "true")
  MLFLOW_TRACKING_URI — MLflow server URI
  S3_BUCKET          — S3 bucket for checkpoints + adapter output
  AWS_*              — AWS credentials
  GPU_PRICE_PER_HOUR — injected by Airflow for cost tracking
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── S3 helpers ────────────────────────────────────────────────────────────────


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _download_s3_file(s3, s3_uri: str, local_path: str) -> str:
    """Download s3://bucket/key to local_path. Returns local_path."""
    if not s3_uri.startswith("s3://"):
        return s3_uri  # already a local path
    without_scheme = s3_uri[5:]
    bucket, key = without_scheme.split("/", 1)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
    s3.download_file(bucket, key, local_path)
    return local_path


def _sync_dir_to_s3(s3, local_dir: str, bucket: str, s3_prefix: str) -> None:
    """Recursively upload local_dir/ to s3://bucket/s3_prefix/."""
    for dirpath, _, filenames in os.walk(local_dir):
        for fname in filenames:
            local_file = os.path.join(dirpath, fname)
            rel = os.path.relpath(local_file, local_dir)
            key = f"{s3_prefix}/{rel}"
            s3.upload_file(local_file, bucket, key)


def _find_latest_s3_checkpoint(
    s3, bucket: str, s3_prefix: str, local_output_dir: str
) -> str | None:
    """
    List checkpoint-N directories under s3_prefix, download the latest one,
    and return its local path. Returns None if no checkpoint exists.
    """
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix + "/checkpoint-")
        objects = resp.get("Contents", [])
        if not objects:
            return None

        # Collect checkpoint step numbers
        checkpoint_steps: set[int] = set()
        for obj in objects:
            after_prefix = obj["Key"][len(s3_prefix):].lstrip("/")
            parts = after_prefix.split("/")
            if parts[0].startswith("checkpoint-"):
                try:
                    step = int(parts[0].split("-")[1])
                    checkpoint_steps.add(step)
                except ValueError:
                    pass

        if not checkpoint_steps:
            return None

        latest_step = max(checkpoint_steps)
        latest_name = f"checkpoint-{latest_step}"
        local_ckpt = os.path.join(local_output_dir, latest_name)
        os.makedirs(local_ckpt, exist_ok=True)

        # Download all files in that checkpoint directory
        paginator = s3.get_paginator("list_objects_v2")
        remote_prefix = f"{s3_prefix}/{latest_name}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=remote_prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(remote_prefix):]
                if not rel:
                    continue
                local_file = os.path.join(local_ckpt, rel)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket, obj["Key"], local_file)

        logger.info("Resumed from S3 checkpoint: %s (step %d)", latest_name, latest_step)
        return local_ckpt

    except Exception as exc:
        logger.warning("Could not load S3 checkpoint: %s", exc)
        return None


# ── Checkpoint callback ───────────────────────────────────────────────────────


def _make_s3_checkpoint_callback(s3, bucket: str, s3_prefix: str):
    """Return a TrainerCallback that syncs each local checkpoint to S3 on save."""
    from transformers import TrainerCallback  # type: ignore[import]

    class S3CheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
            ckpt_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )
            if os.path.isdir(ckpt_dir):
                remote = f"{s3_prefix}/checkpoint-{state.global_step}"
                try:
                    _sync_dir_to_s3(s3, ckpt_dir, bucket, remote)
                    logger.info(
                        "Checkpoint synced to S3: checkpoint-%d", state.global_step
                    )
                except Exception as exc:
                    logger.warning("S3 checkpoint sync failed: %s", exc)

    return S3CheckpointCallback()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import torch  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import]
    from datasets import load_dataset  # type: ignore[import]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import]

    # ── Config from environment ──────────────────────────────────────────────
    run_id = os.getenv("LLM_TRAIN_RUN_ID") or f"llm-{int(time.time())}"
    model_id = os.environ["LLM_MODEL_ID"]
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    dataset_s3 = os.environ["TRAIN_DATASET_S3"]

    max_seq_len = int(os.getenv("MAX_SEQ_LEN", "2048"))
    max_steps = int(os.getenv("MAX_STEPS", "500"))
    save_steps = int(os.getenv("SAVE_STEPS", "100"))
    batch_size = int(os.getenv("PER_DEVICE_BATCH_SIZE", "1"))
    grad_accum = int(os.getenv("GRADIENT_ACCUM_STEPS", "4"))
    learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))

    lora_enabled = os.getenv("LORA_ENABLED", "true").lower() == "true"
    lora_rank = int(os.getenv("LORA_RANK", "8"))

    use_deepspeed = os.getenv("USE_DEEPSPEED", "true").lower() == "true"
    ds_stage = int(os.getenv("DEEPSPEED_STAGE", "3"))

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    bucket = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
    gpu_price = float(os.getenv("GPU_PRICE_PER_HOUR", "0"))
    is_spot = os.getenv("USE_SPOT", "false").lower() == "true"

    output_dir = f"/tmp/llm_checkpoints/{run_id}"
    s3_ckpt_prefix = f"llm-training/{run_id}/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    job_start = time.time()
    s3 = _s3_client()

    logger.info(
        "LLM fine-tuning start: run_id=%s model=%s lora=%s deepspeed=%s stage=%d",
        run_id, model_id, lora_enabled, use_deepspeed, ds_stage,
    )

    # ── Download JSONL dataset from S3 ───────────────────────────────────────
    local_jsonl = f"/tmp/{run_id}_train.jsonl"
    _download_s3_file(s3, dataset_s3, local_jsonl)

    # ── Resume from S3 checkpoint if available ───────────────────────────────
    resume_from = _find_latest_s3_checkpoint(s3, bucket, s3_ckpt_prefix, output_dir)

    # ── Load tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Load base model ──────────────────────────────────────────────────────
    # device_map="auto" causes conflicts with DeepSpeed; use None and let
    # DeepSpeed handle placement. For non-DeepSpeed, "auto" is fine.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=None if use_deepspeed else "auto",
    )

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    if lora_enabled:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────────
    # load_dataset with streaming=False: full JSONL loaded into RAM.
    # For datasets > a few GB, set streaming=True (pass IterableDataset to SFTTrainer).
    dataset = load_dataset("json", data_files=local_jsonl, split="train")
    logger.info("Dataset loaded: %d examples", len(dataset))

    # ── DeepSpeed config ─────────────────────────────────────────────────────
    ds_config: dict | None = None
    if use_deepspeed:
        ds_config = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": ds_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                # Stage 3: partition params, grads, optimizer states
                **(
                    {
                        "stage3_max_live_parameters": 1e9,
                        "stage3_max_reuse_distance": 1e9,
                        "stage3_prefetch_bucket_size": "auto",
                        "stage3_param_persistence_threshold": "auto",
                        "reduce_bucket_size": "auto",
                    }
                    if ds_stage == 3
                    else {}
                ),
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 50,
            "wall_clock_breakdown": False,
        }

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=max_seq_len,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=10,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        bf16=True,
        deepspeed=ds_config,
        resume_from_checkpoint=resume_from,
        report_to="none",  # MLflow logged manually below
        dataset_text_field="text",  # expected field in JSONL records
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [_make_s3_checkpoint_callback(s3, bucket, s3_ckpt_prefix)]

    # ── MLflow run ───────────────────────────────────────────────────────────
    mlflow_active = bool(mlflow_uri)
    if mlflow_active:
        import mlflow  # type: ignore[import]
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.start_run(run_name=run_id)
        mlflow.log_params({
            "model_id": model_id,
            "max_seq_len": max_seq_len,
            "max_steps": max_steps,
            "lora_rank": lora_rank if lora_enabled else None,
            "lora_enabled": lora_enabled,
            "deepspeed_stage": ds_stage if use_deepspeed else None,
            "use_deepspeed": use_deepspeed,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "learning_rate": learning_rate,
        })

    # ── Train ────────────────────────────────────────────────────────────────
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
        trainer.train()
    except Exception:
        if mlflow_active:
            import mlflow  # type: ignore[import]
            mlflow.end_run(status="FAILED")
        raise

    # ── Save final adapter to S3 ─────────────────────────────────────────────
    final_local = os.path.join(output_dir, "final")
    trainer.save_model(final_local)
    tokenizer.save_pretrained(final_local)

    # S3 path: llm-adapters/{model-id-slug}/{run_id}/
    model_slug = model_id.replace("/", "-").replace(".", "-").lower()
    adapter_s3_prefix = f"llm-adapters/{model_slug}/{run_id}"
    _sync_dir_to_s3(s3, final_local, bucket, adapter_s3_prefix)
    adapter_s3_path = f"s3://{bucket}/{adapter_s3_prefix}"
    logger.info("Adapter saved to: %s", adapter_s3_path)

    # ── MLflow final logging ─────────────────────────────────────────────────
    if mlflow_active:
        import mlflow  # type: ignore[import]
        elapsed_hours = (time.time() - job_start) / 3600
        mlflow.set_tag("adapter_s3_path", adapter_s3_path)
        mlflow.set_tag("llm_model_id", model_id)
        mlflow.set_tag("llm_train_run_id", run_id)
        if gpu_price > 0:
            mlflow.set_tag(
                "estimated_cost_usd", round(gpu_price * elapsed_hours, 4)
            )
            mlflow.set_tag(
                "instance_type", "spot" if is_spot else "on_demand"
            )
        mlflow.end_run()

    logger.info(
        "Training complete. Steps=%d, elapsed=%.1fh, adapter=%s",
        max_steps,
        (time.time() - job_start) / 3600,
        adapter_s3_path,
    )


if __name__ == "__main__":
    main()
