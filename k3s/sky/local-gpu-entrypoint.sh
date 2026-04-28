#!/bin/bash
# Entrypoint for local GPU SSH node used by SkyPilot SSH Node Pools.
# It injects the SSH authorized key and then boots init (systemd) so the node
# can run k3s bootstrap steps during `sky ssh up`.
set -euo pipefail

if [ -n "${SSH_AUTHORIZED_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "${SSH_AUTHORIZED_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
fi

mkdir -p /run/sshd

# GPU Operator operands use HostPath mounts that require shared/slave
# propagation. In this nested Docker->k3s setup mounts are often private.
ensure_mount_propagation() {
    local target="$1"
    local current
    current="$(findmnt -n -o PROPAGATION "${target}" 2>/dev/null || true)"
    case "${current}" in
        shared*|rshared*|slave*|rslave*)
            return 0
            ;;
    esac

    mount --make-rshared "${target}" 2>/dev/null || \
        mount --make-rslave "${target}" 2>/dev/null || true
}

# Must run before k3s starts so kubelet can create NVIDIA operand pods.
ensure_mount_propagation /
ensure_mount_propagation /run

# WSL GPU may not expose NVIDIA PCI vendor via NFD.
# Seed the label k3s should apply at node registration so GPU Operator can
# discover this node as GPU-capable.
K3S_CONFIG="/etc/rancher/k3s/config.yaml"
K3S_GPU_LABEL="feature.node.kubernetes.io/pci-10de.present=true"
mkdir -p "$(dirname "${K3S_CONFIG}")"
if [ ! -f "${K3S_CONFIG}" ]; then
    printf 'snapshotter: native\n' > "${K3S_CONFIG}"
fi
if ! grep -qF "${K3S_GPU_LABEL}" "${K3S_CONFIG}"; then
    {
        echo "node-label:"
        echo "  - ${K3S_GPU_LABEL}"
    } >> "${K3S_CONFIG}"
fi

# k3s kubelet requires the pids controller to be mounted in this nested
# container setup. Docker Desktop sometimes leaves /sys/fs/cgroup/pids
# unmounted even with --privileged, so mount it proactively.
if [ -d /sys/fs/cgroup/pids ] && ! grep -qE '[[:space:]]/sys/fs/cgroup/pids[[:space:]]' /proc/mounts; then
    mount -t cgroup -o pids cgroup /sys/fs/cgroup/pids || true
fi

if [ -x /sbin/init ]; then
    exec /sbin/init
fi

# Fallback for images without init.
exec /usr/sbin/sshd -D -e
