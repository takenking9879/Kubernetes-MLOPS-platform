"""
app/backend/routers/model_architectures.py

Model architecture browser and upload endpoint (Phase 8).

Endpoints:
  GET  /api/v2/model-architectures         — list built-in + user-uploaded architectures
  POST /api/v2/model-architectures/upload  — validate + upload a custom nn.Module .py file
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2/model-architectures", tags=["model-architectures"])

_S3_BUCKET = os.getenv("S3_BUCKET", "k8s-mlops-platform-bucket")
_S3_PREFIX = "model-architectures"

# ── Built-in architectures (always returned) ──────────────────────────────────

_BUILTIN: list[dict] = [
    {
        "id": "ann",
        "name": "MLP (ANN)",
        "description": "Multi-layer perceptron for tabular data classification / regression.",
        "builtin": True,
    },
    {
        "id": "ssm",
        "name": "SSM Tabular",
        "description": "State Space Model treating tabular features as a 1D sequence (S4-style).",
        "builtin": True,
    },
    {
        "id": "bae",
        "name": "BAE Ensemble",
        "description": "Bootstrap Aggregating Ensemble of autoencoders for anomaly / classification.",
        "builtin": True,
    },
    {
        "id": "xgboost",
        "name": "XGBoost",
        "description": "Gradient boosted trees via XGBoost — CPU training, GPU tree_method available.",
        "builtin": True,
    },
]


# ── Response models ───────────────────────────────────────────────────────────

class ArchitectureInfo(BaseModel):
    id: str
    name: str
    description: str
    builtin: bool
    s3_path: str | None = None
    uploaded_at: str | None = None


class UploadResult(BaseModel):
    id: str
    name: str
    s3_path: str
    status: str                # "validated"


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_architecture(source: str, filename: str) -> None:
    """
    Parse AST to confirm the file:
    - Is valid Python
    - Defines at least one class that appears to inherit nn.Module
    - That class has a forward() method

    Raises HTTPException(422) if validation fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Python syntax error in {filename}: {exc}",
        )

    # Find classes that inherit from something containing "Module" or "nn.Module"
    module_classes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = f"{getattr(base.value, 'id', '')}.{base.attr}"
            if "Module" in base_name:
                module_classes.append(node.name)

    if not module_classes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{filename}: no class inheriting from nn.Module found. "
                "The file must define a PyTorch model class, e.g.:\n"
                "  class MyModel(nn.Module): ..."
            ),
        )

    # Check that at least one module class has a forward() method
    has_forward = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in module_classes:
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "forward":
                has_forward = True
                break

    if not has_forward:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{filename}: class(es) {module_classes} must implement a forward() method."
            ),
        )


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_client():
    import boto3  # type: ignore[import]
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _list_uploaded() -> list[dict]:
    """List user-uploaded architectures from S3 metadata index."""
    try:
        s3 = _s3_client()
        resp = s3.get_object(Bucket=_S3_BUCKET, Key=f"{_S3_PREFIX}/index.json")
        return json.loads(resp["Body"].read())
    except Exception:
        return []


def _save_uploaded(entry: dict) -> None:
    """Append entry to the S3 metadata index."""
    try:
        s3 = _s3_client()
        existing = _list_uploaded()
        existing = [e for e in existing if e.get("id") != entry["id"]]
        existing.append(entry)
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=f"{_S3_PREFIX}/index.json",
            Body=json.dumps(existing).encode(),
            ContentType="application/json",
        )
    except Exception:
        pass  # index update failure is non-fatal


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/", response_model=list[ArchitectureInfo])
async def list_architectures() -> list[ArchitectureInfo]:
    """Return built-in architectures plus any user-uploaded ones."""
    result = [ArchitectureInfo(**a) for a in _BUILTIN]
    for entry in _list_uploaded():
        result.append(
            ArchitectureInfo(
                id=entry.get("id", ""),
                name=entry.get("name", ""),
                description=entry.get("description", ""),
                builtin=False,
                s3_path=entry.get("s3_path"),
                uploaded_at=entry.get("uploaded_at"),
            )
        )
    return result


@router.post("/upload", response_model=UploadResult)
async def upload_architecture(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(default=""),
) -> UploadResult:
    """
    Validate + upload a custom PyTorch model architecture (.py file).

    Validates Python syntax and checks that the file defines a class
    inheriting nn.Module with a forward() method.

    The file is stored at s3://{bucket}/model-architectures/{upload_id}/{name}.py.
    """
    filename = file.filename or "model.py"
    if not filename.endswith(".py"):
        raise HTTPException(
            status_code=422, detail="Only Python (.py) files are accepted."
        )

    content = await file.read()
    if len(content) > 512 * 1024:
        raise HTTPException(
            status_code=413, detail="File too large (max 512 KB)."
        )

    source = content.decode("utf-8", errors="replace")
    _validate_architecture(source, filename)

    # Generate a stable ID from name
    arch_id = "user_" + hashlib.sha256(name.encode()).hexdigest()[:8]
    upload_id = hashlib.sha256(content).hexdigest()[:12]
    s3_key = f"{_S3_PREFIX}/{arch_id}/{upload_id}/{filename}"

    try:
        s3 = _s3_client()
        s3.put_object(
            Bucket=_S3_BUCKET,
            Key=s3_key,
            Body=content,
            ContentType="text/x-python",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"S3 upload failed: {exc}",
        )

    s3_path = f"s3://{_S3_BUCKET}/{s3_key}"
    entry = {
        "id": arch_id,
        "name": name,
        "description": description or f"User-uploaded model: {filename}",
        "s3_path": s3_path,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_uploaded(entry)

    return UploadResult(
        id=arch_id,
        name=name,
        s3_path=s3_path,
        status="validated",
    )
