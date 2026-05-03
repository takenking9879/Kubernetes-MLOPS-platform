#!/usr/bin/env bash
# k3s on WSL2 only: sync Windows hosts + portproxy rules so *.localhost reaches
# the ingress controller. Not needed with Docker Desktop / kubeadm.
#
# - Writes *.localhost → WSL2 IP in C:\Windows\System32\drivers\etc\hosts
# - Sets netsh portproxy: 127.0.0.1:80/443 → WSL2 IP:80/443
#   (Chrome ignores hosts for *.localhost and hardroutes to 127.0.0.1)
#
# Idempotent. Triggers a single UAC prompt when direct write is blocked.

set -euo pipefail

WIN_HOSTS="/mnt/c/Windows/System32/drivers/etc/hosts"
MARKER_START="# k3s MLOps Platform --- BEGIN"
MARKER_END="# k3s MLOps Platform --- END"

log()  { echo "[update-win-hosts] $*"; }
warn() { echo "[update-win-hosts] WARNING: $*" >&2; }

WSL_IP="${1:-}"
if [[ -z "${WSL_IP}" ]]; then
  WSL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [[ -z "${WSL_IP}" ]]; then
  warn "Could not determine WSL2 IP. Skipping."
  exit 0
fi

if [[ ! -f "${WIN_HOSTS}" ]]; then
  warn "Windows hosts file not found at ${WIN_HOSTS}."
  exit 0
fi

# Resolve powershell.exe — not always in PATH when called from deploy.sh / sudo.
POWERSHELL="$(command -v powershell.exe 2>/dev/null \
  || echo /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe)"
if [[ ! -x "${POWERSHELL}" ]]; then
  warn "powershell.exe not found. Skipping."
  exit 0
fi

# Build new hosts content: strip old block, append fresh one.
build_new_hosts() {
  local src="$1"
  awk -v s="${MARKER_START}" -v e="${MARKER_END}" '
    $0 == s { skip=1; next }
    $0 == e { skip=0; next }
    !skip   { print }
  ' "${src}"
  printf '\n%s\n%s  airflow.localhost\n%s  app.localhost\n%s  mlflow.localhost\n%s  grafana.localhost\n%s  kafka.localhost\n%s  serving.localhost\n%s  prom-write.localhost\n%s\n' \
    "${MARKER_START}" \
    "${WSL_IP}" "${WSL_IP}" "${WSL_IP}" "${WSL_IP}" "${WSL_IP}" "${WSL_IP}" "${WSL_IP}" \
    "${MARKER_END}"
}

# ── Build PS1 that runs elevated (hosts + portproxy) ──────────────────────────
WIN_TEMP="$("${POWERSHELL}" -NoProfile -NonInteractive -Command \
  '[System.IO.Path]::GetTempPath()' 2>/dev/null | tr -d '\r\n')"
WSL_TEMP="$(wslpath "${WIN_TEMP}")"

CONTENT_FILE="${WSL_TEMP}k3s-hosts-$$.txt"
WIN_CONTENT="$(wslpath -w "${CONTENT_FILE}")"
PS1_FILE="${WSL_TEMP}update-win-hosts-$$.ps1"
WIN_PS1="$(wslpath -w "${PS1_FILE}")"

build_new_hosts "${WIN_HOSTS}" > "${CONTENT_FILE}"

cat > "${PS1_FILE}" << PSEOF
param([string]\$src, [string]\$dst, [string]\$wslIp)
# Update hosts file
Copy-Item -Path \$src -Destination \$dst -Force
# Update portproxy (remove stale rules on 80/443, add fresh ones)
netsh interface portproxy delete v4tov4 listenport=80  listenaddress=127.0.0.1 2>\$null
netsh interface portproxy delete v4tov4 listenport=443 listenaddress=127.0.0.1 2>\$null
netsh interface portproxy add    v4tov4 listenport=80  listenaddress=127.0.0.1 connectport=80  connectaddress=\$wslIp
netsh interface portproxy add    v4tov4 listenport=443 listenaddress=127.0.0.1 connectport=443 connectaddress=\$wslIp
PSEOF

log "Launching elevated PowerShell for hosts + portproxy (UAC prompt will appear)..."

"${POWERSHELL}" -NoProfile -NonInteractive -Command \
  "Start-Process powershell -Verb RunAs -Wait \
   -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"${WIN_PS1}\" \"${WIN_CONTENT}\" \"C:\Windows\System32\drivers\etc\hosts\" \"${WSL_IP}\"'"

rm -f "${CONTENT_FILE}" "${PS1_FILE}"

# Verify
if grep -q "${MARKER_START}" "${WIN_HOSTS}" 2>/dev/null; then
  log "Hosts updated: *.localhost → ${WSL_IP}"
else
  warn "Hosts verify failed — UAC may have been denied."
fi

PROXY_CHECK="$("${POWERSHELL}" -NoProfile -Command \
  "netsh interface portproxy show v4tov4" 2>/dev/null | grep -c "127.0.0.1" || true)"
if [[ "${PROXY_CHECK}" -ge 2 ]]; then
  log "Portproxy active: 127.0.0.1:80/443 → ${WSL_IP}:80/443"
else
  warn "Portproxy verify failed."
fi
