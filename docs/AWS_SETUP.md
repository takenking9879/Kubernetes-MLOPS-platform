# AWS Setup for Multi-Node GPU Training

This document covers the AWS IAM configuration needed for multi-node GPU training
via SkyPilot (`ray-gpu-multinode-aws.yaml`).  The goal is to avoid embedding
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in SkyPilot YAMLs by using an IAM
instance profile instead.

---

## IAM Role: `skypilot-training-role`

### 1. Create the role

```bash
aws iam create-role \
  --role-name skypilot-training-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }]
  }'
```

### 2. Attach S3 permissions

Create an inline policy (`skypilot-s3-policy.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::k8s-mlops-platform-bucket",
        "arn:aws:s3:::k8s-mlops-platform-bucket/*"
      ]
    },
    {
      "Sid": "MLflowArtifacts",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::k8s-mlops-platform-bucket/mlflow-artifacts/*"
      ]
    }
  ]
}
```

```bash
aws iam put-role-policy \
  --role-name skypilot-training-role \
  --policy-name skypilot-s3-policy \
  --policy-document file://skypilot-s3-policy.json
```

### 3. Create the instance profile

```bash
aws iam create-instance-profile \
  --instance-profile-name skypilot-training-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name skypilot-training-profile \
  --role-name skypilot-training-role
```

---

## SkyPilot config: `~/.sky/config.yaml`

Tell SkyPilot to attach the instance profile to all AWS instances and set the
jobs controller disk size below RunPod's 40 GB limit:

```yaml
aws:
  instance_profile: skypilot-training-profile   # attached to every training VM

jobs:
  controller:
    resources:
      disk_size: 30                              # RunPod cap is 40 GB
```

With this in place, **remove `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
from all SkyPilot YAML `envs:` sections** — the instance profile provides
credentials automatically.

---

## EFA (Elastic Fabric Adapter)

EFA is enabled automatically on p4d / p3dn instances.  The multi-node YAML
sets the required environment variables:

| Variable | Value | Purpose |
|---|---|---|
| `NCCL_IB_DISABLE` | `0` | Enable InfiniBand / EFA in NCCL |
| `NCCL_SOCKET_IFNAME` | `eth0` | Primary network interface |
| `FI_EFA_USE_DEVICE_RDMA` | `1` | Enable EFA RDMA for p4d |
| `RDMAV_FORK_SAFE` | `1` | Required for EFA + multiprocessing |
| `NCCL_DEBUG` | `INFO` | Log EFA init for troubleshooting |

Confirm EFA is negotiated by checking logs for:
```
NCCL INFO Transport EFA initialized
```

---

## Spot availability

`p4d.24xlarge` spot is limited.  Recommended test sequence:

1. **Validate multi-node Ray**: use `p3.8xlarge` (4×V100, no EFA).  Cheapest
   path to confirm Ray head/worker join logic works.
2. **Validate EFA**: use `p3dn.24xlarge` spot (8×V100, 100 Gbps EFA).
   Widely available in `us-east-1`.
3. **Full throughput**: `p4d.24xlarge` spot (8×A100-40GB, 400 Gbps EFA).
   Reserve capacity if spot is unavailable.

The `any_of` list in `ray-gpu-multinode-aws.yaml` already encodes this fallback
order — SkyPilot picks the first available option.

---

## Minimal permissions for the Airflow worker

The Airflow pod (which calls `sky.jobs.launch()`) needs AWS credentials to
provision EC2 instances.  Minimum IAM policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "iam:PassRole",
        "iam:GetInstanceProfile"
      ],
      "Resource": "*"
    }
  ]
}
```

Provide these as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment
variables on the Airflow worker — **not** in any checked-in YAML file.
