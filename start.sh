#!/bin/bash
# =============================================================================
# Khedut Voice AI -- Headless Startup Wrapper for Raspberry Pi
# =============================================================================
# Called by systemd. Handles:
#   1. Wait for internet (infinite retry, no timeout)
#   2. Validate GEMINI_API_KEY is set
#   3. Activate venv and launch the voice agent
#
# Logs visible via:  journalctl -u khedut-voice-pi -f
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "=============================================="
log "  Khedut Voice AI -- Startup"
log "=============================================="

# -- Audio & PulseAudio environment for headless systemd service --------------
export USER="${USER:-pi}"
export HOME="${HOME:-/home/$USER}"
PI_UID="$(id -u $USER 2>/dev/null || echo 1000)"
export XDG_RUNTIME_DIR="/run/user/$PI_UID"
export PULSE_SERVER="unix:$XDG_RUNTIME_DIR/pulse/native"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"

# -- 1. Load .env --------------------------------------------------------------
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -o allexport
    source "$SCRIPT_DIR/.env"
    set +o allexport
    log ".env loaded."
else
    log "WARNING: .env file not found at $SCRIPT_DIR/.env"
fi

# -- 2. Wait for internet (ping Google DNS, retry every 5s, no limit) ----------
log "Waiting for internet connection..."
ATTEMPT=0
while ! ping -c 1 -W 3 8.8.8.8 > /dev/null 2>&1; do
    ATTEMPT=$((ATTEMPT + 1))
    log "  [Attempt $ATTEMPT] No internet yet... retrying in 5s"
    sleep 5
done
log "Internet OK after $ATTEMPT retries."

# -- 3. Pull latest code from Git ----------------------------------------------
log "Pulling latest code from repository..."
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    if git pull; then
        log "Git pull successful (up to date)."
    else
        log "WARNING: git pull failed. Continuing with local code."
    fi
fi

# -- 4. Validate GEMINI_API_KEY ------------------------------------------------
CLEAN_KEY="$(echo "$GEMINI_API_KEY" | tr -d "\"' ")"
if [ -z "$CLEAN_KEY" ]; then
    log "ERROR: GEMINI_API_KEY is not set in .env"
    log "       Edit $SCRIPT_DIR/.env and add:"
    log "       GEMINI_API_KEY=AIzaSy..."
    log "       Sleeping 60s before exit so systemd does not spin-loop."
    sleep 60
    exit 1
fi
log "GEMINI_API_KEY found (${#CLEAN_KEY} chars)."

# -- 5. Activate virtual environment ------------------------------------------
VENV="$SCRIPT_DIR/venv/bin/activate"
if [ ! -f "$VENV" ]; then
    log "ERROR: venv not found at $VENV"
    log "       Run pi_setup.sh first."
    sleep 60
    exit 1
fi
source "$VENV"
log "venv activated: $(python --version)"

# -- 6. Launch voice agent (exec = systemd tracks this PID directly) -----------
log "Allowing audio drivers to settle..."
sleep 2
log "Launching pi_voice_agent.py ..."
log "=============================================="

exec python "$SCRIPT_DIR/pi_voice_agent.py"
