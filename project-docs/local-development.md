# Local Development

## Status
- `working`: repeatable local scripts for up/down/deploy + GPU runtime setup in k3s/WSL2

## Design
- `up-k3s-gpu.sh` configures k3s runtime and GPU plugin path.
- `deploy.sh` applies namespaces, secrets, and selected platform components.
- `down-k3s-gpu.sh` tears down runtime while preserving useful local state.
- `update-win-hosts.sh` handles WSL2 ingress accessibility constraints.

## Why k3s
Local GPU-aware experimentation is explicitly prioritized. The scripts and docs reflect that runtime and ingress behavior were engineered for iterative debugging.

## Trade-Offs
- Script-centric workflows are fast to iterate but less declarative than full GitOps.
- Local environment drift is possible without stricter reconciliation controls.

## Evidence Pointers
- `k3s/up-k3s-gpu.sh`
- `k3s/down-k3s-gpu.sh`
- `k3s/deploy.sh`
- `k3s/README-k3s-gpu.md`
