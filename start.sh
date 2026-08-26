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

# -- 3. Validate GEMINI_API_KEY ------------------------------------------------
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

# -- 4. Activate virtual environment ------------------------------------------
VENV="$SCRIPT_DIR/venv/bin/activate"
if [ ! -f "$VENV" ]; then
    log "ERROR: venv not found at $VENV"
    log "       Run pi_setup.sh first."
    sleep 60
    exit 1
fi
source "$VENV"
log "venv activated: $(python --version)"

# -- 5. Launch voice agent (exec = systemd tracks this PID directly) -----------
log "Launching pi_voice_agent.py ..."
log "=============================================="

exec python "$SCRIPT_DIR/pi_voice_agent.py"
