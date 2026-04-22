#!/bin/bash
# Entrypoint for the local GPU SSH node container.
# Injects the SSH authorized key from SSH_AUTHORIZED_KEY env var, starts sshd,
# then keeps the container alive for SkyPilot to SSH into.
set -e

if [ -n "${SSH_AUTHORIZED_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${SSH_AUTHORIZED_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
fi

/usr/sbin/sshd -D &

exec tail -f /dev/null
