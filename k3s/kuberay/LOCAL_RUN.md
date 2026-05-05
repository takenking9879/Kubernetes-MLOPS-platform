# Local Pipeline Run

## Prerequisites

- Conda env `ray-experiments` activated
- Ray cluster up locally: `ray start --head` (or `RAY_ADDRESS=local` for single-node)
- S3 credentials exported (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Params YAML at S3 or local path

## Run

```bash
conda activate ray-experiments

# Load S3/Iceberg creds from root .env
python -c "from dotenv import load_dotenv; load_dotenv()"  # verify alone
# Or source inline:
eval $(python -c "
from dotenv import load_dotenv; load_dotenv()
import os
for k,v in os.environ.items():
    if k.startswith('AWS_') or k in ('MLFLOW_TRACKING_URI',):
        print(f'export {k}=\"{v}\"')
")

ray start --head --disable-usage-stats 2>/dev/null || true

export PARAMS_S3_PATH="s3://k8s-mlops-platform-bucket/runs/training/train-network_traffic-235537-20260503T213841Z-f549a8/params_training.yaml"
export AWS_REGION="${AWS_REGION:-us-east-2}"
export MLFLOW_TRACKING_URI=""                    # skip MLflow for local test
export TRAIN_RUN_ID="local-test-$(date +%s)"
export USE_GPU="${USE_GPU:-auto}"
export REPORT_FREQUENCY="${REPORT_FREQUENCY:-5}"
export OUTPUT_DIR="/tmp/models"                  # local dir instead of S3

python k3s/kuberay/main.py

# Cleanup
ray stop 2>/dev/null || true
```

## Key env vars

| Var | Default | Notes |
|-----|---------|-------|
| `PARAMS_S3_PATH` | — | S3 URI to `params_training.yaml` |
| `PARAMS_PATH` | — | Local path (overrides S3) |
| `AWS_ACCESS_KEY_ID` | — | Required for S3 + Iceberg |
| `AWS_SECRET_ACCESS_KEY` | — | Required for S3 + Iceberg |
| `AWS_REGION` | `us-east-2` | |
| `MLFLOW_TRACKING_URI` | — | Empty = skip MLflow |
| `TRAIN_RUN_ID` | from YAML | Unique run identifier |
| `USE_GPU` | auto | `true`/`false`/`auto` |
| `CPUS_PER_WORKER` | auto | CPU threads per worker |
| `REPORT_FREQUENCY` | `5` | Epochs between reports |
| `OUTPUT_DIR` | `s3://.../v1/models` | Model save path |

## Quick syntax check (no S3/Ray needed)

```bash
python -c "from src.pipeline.utils.pytorch_utils import train_func; print('✓ import OK')"
python -c "from src.pipeline.base_trainer import BaseTrainer; print('✓ import OK')"
```
