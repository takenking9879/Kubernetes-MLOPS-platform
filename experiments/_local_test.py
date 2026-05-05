"""Local test wrapper — monkey-patches os.path.join for local paths, then runs main."""
import os, sys, time, tempfile, urllib.parse

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Download params from S3 and inject local dsl_path ---
import boto3, yaml
s3_uri = os.environ.get(
    "PARAMS_S3_PATH",
    "s3://k8s-mlops-platform-bucket/runs/training/train-network_traffic-235537-20260503T213841Z-f549a8/params_training.yaml",
)
parsed = urllib.parse.urlparse(s3_uri)
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-2"),
)
tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
s3.download_file(parsed.netloc, parsed.path.lstrip("/"), tmp.name)
with open(tmp.name) as f:
    params = yaml.safe_load(f)
os.unlink(tmp.name)

# Inject dsl_path so os.path.join('/home/ray/', dsl_path.lstrip('/'))
# resolves to the local DSL file when /home/ray/ is patched → REPO_ROOT.
params.setdefault("spark", {}).setdefault("preprocessing", {})[
    "dsl_path"
] = "k3s/spark/preprocess/dsl_001.yaml"

# Disable MLflow — ngrok tunnel is offline for local testing.
params.get("kuberay", {}).get("model", {}).pop("mlflow_tracking_uri", None)

patched = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
yaml.dump(params, patched)
patched.close()
os.environ["PARAMS_PATH"] = patched.name
os.environ.pop("PARAMS_S3_PATH", None)  # avoid conflict

# --- Monkey-patch os.path.join: redirect /home/ray/ → REPO_ROOT ---
_original_join = os.path.join
def _patched_join(a, *p):
    if isinstance(a, str) and a == '/home/ray/':
        a = REPO_ROOT + '/'
    return _original_join(a, *p)
os.path.join = _patched_join

os.environ["MLFLOW_TRACKING_URI"] = ""  # override .env value
os.environ["TRAIN_RUN_ID"] = f"local-test-{int(time.time())}"
os.environ["USE_GPU"] = "false"  # CPU for quick test
os.environ["REPORT_FREQUENCY"] = "5"
os.environ["OUTPUT_DIR"] = "/tmp/models"
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"

# Fractional GPU for 2 workers, full GPU for 1 worker
_USE_GPU_ENV = os.environ.get("USE_GPU", "auto").lower()
if _USE_GPU_ENV in ("true", "1"):
    import torch as _torch
    if _torch.cuda.is_available():
        nw = int(os.environ.get("NUM_WORKERS", "1"))
        if nw >= 2:
            os.environ["GPUS_PER_WORKER"] = "0.5"
            os.environ["NUM_WORKERS"] = str(nw)
            print(f"[local_test] GPU detected ({_torch.cuda.get_device_name(0)}), "
                  f"{nw} workers, fractional GPU=0.5")
        else:
            os.environ["GPUS_PER_WORKER"] = "1"
            print(f"[local_test] GPU detected ({_torch.cuda.get_device_name(0)}), "
                  f"1 worker, full GPU=1")
    else:
        os.environ["USE_GPU"] = "false"
        print("[local_test] No GPU detected, falling back to CPU")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Some internal imports use "pipeline." not "src.pipeline."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

print(f"[local_test] dsl_path → k3s/spark/preprocess/dsl_001.yaml")
print(f"[local_test] mlflow_tracking_uri removed: {params.get('kuberay', {}).get('model', {}).get('mlflow_tracking_uri', 'NOT_PRESENT')}")
print(f"[local_test] patched params at: {patched.name}")
print("=" * 60, flush=True)

import runpy

# Ray workers are separate processes that need AWS creds (Iceberg catalog)
# and PYTHONPATH (pipeline imports). Forward them via runtime_env.
import ray as _ray
_original_ray_init = _ray.init
def _patched_ray_init(*args, **kwargs):
    py_path = REPO_ROOT + "/src:" + REPO_ROOT
    re_kwargs = kwargs.get("runtime_env")
    if re_kwargs is None:
        re_kwargs = {}
        kwargs["runtime_env"] = re_kwargs
    if isinstance(re_kwargs, dict):
        env_vars = re_kwargs.setdefault("env_vars", {})
        env_vars.setdefault("PYTHONPATH", py_path)
        for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
            _v = os.environ.get(_k)
            if _v:
                env_vars.setdefault(_k, _v)
    return _original_ray_init(*args, **kwargs)
_ray.init = _patched_ray_init

runpy.run_path(os.path.join(REPO_ROOT, "k3s/kuberay/main.py"), run_name="__main__")
