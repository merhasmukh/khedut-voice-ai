#!/bin/bash
# =============================================================================
# Khedut Voice AI -- Raspberry Pi Setup Script
# =============================================================================
# Compatible with standard Raspberry Pi OS (Python 3.9+)
# Installs system audio packages, creates virtualenv, and installs minimal
# voice AI dependencies.
#
# Usage:  chmod +x pi_setup.sh && ./pi_setup.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Khedut Voice AI -- Raspberry Pi Setup 🍓"
echo "=============================================="
echo ""

# -- 1. System packages -------------------------------------------------------
echo "[1/4] Installing system audio packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libportaudio2 \
    libsdl2-dev \
    libsdl2-mixer-2.0-0 \
    pulseaudio \
    pulseaudio-module-bluetooth \
    bluez \
    ffmpeg \
    alsa-utils
echo "      System packages installed."
echo ""

# -- 2. Bluetooth speaker guide -----------------------------------------------
echo "=============================================="
echo "  Bluetooth Speaker Pairing Guide"
echo "=============================================="
echo "  To pair a Bluetooth speaker, run:"
echo "    bluetoothctl"
echo "    > power on"
echo "    > scan on"
echo "    > pair   XX:XX:XX:XX:XX:XX   (your speaker MAC)"
echo "    > connect XX:XX:XX:XX:XX:XX"
echo "    > trust  XX:XX:XX:XX:XX:XX"
echo "    > exit"
echo ""
echo "  Then set as default audio output:"
echo "    pactl set-default-sink bluez_sink.XX_XX_XX_XX_XX_XX.a2dp_sink"
echo ""

# -- 3. Python virtual environment --------------------------------------------
echo "[2/4] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "      Created new venv with $(python3 --version)."
else
    echo "      Found existing venv ($(venv/bin/python --version))."
fi

source venv/bin/activate
pip install --upgrade pip --quiet
echo "      Active Python version: $(python --version)"
echo ""

# -- 4. Install dependencies --------------------------------------------------
echo "[3/4] Installing lightweight Voice AI dependencies (requirements-pi.txt)..."
pip install -r requirements-pi.txt --quiet
echo "      Dependencies installed."
echo ""

# -- 5. Initialize SQLite Database -------------------------------------------
echo "[4/5] Initializing SQLite database (khedut_voice.db)..."
python -c "
import asyncio
from database.connection import init_db
asyncio.run(init_db())
"
echo ""

# -- 6. Check .env ------------------------------------------------------------
echo "[5/5] Checking .env configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "      Created .env from .env.example"
        echo "      *** Please edit .env and add your GEMINI_API_KEY ***"
    fi
else
    if grep -q "GEMINI_API_KEY=" .env; then
        echo "      .env found with GEMINI_API_KEY."
    else
        echo "      WARNING: GEMINI_API_KEY not found in .env"
    fi
fi
echo ""

# -- Audio devices check ------------------------------------------------------
echo "Available audio devices:"
python -c "
import pyaudio
pa = pyaudio.PyAudio()
count = pa.get_device_count()
print(f'  {count} devices found:')
for i in range(count):
    info = pa.get_device_info_by_index(i)
    tags = []
    if info['maxInputChannels'] > 0:
        tags.append('INPUT')
    if info['maxOutputChannels'] > 0:
        tags.append('OUTPUT')
    if tags:
        print(f'    [{i}] {info[\"name\"]} ({\" | \".join(tags)})')
try:
    di = pa.get_default_input_device_info()
    print(f'  Default Input  -> [{di[\"index\"]}] {di[\"name\"]}')
except:
    print('  Default Input  -> (none found)')
try:
    do = pa.get_default_output_device_info()
    print(f'  Default Output -> [{do[\"index\"]}] {do[\"name\"]}')
except:
    print('  Default Output -> (none found)')
pa.terminate()
"
echo ""

# -- Done ---------------------------------------------------------------------
echo "==============================================" 
echo "  Setup Complete! 🚀"
echo "=============================================="
echo ""

# -- 6. Install systemd auto-start service ------------------------------------
echo "[6/6] Installing systemd auto-start service..."
chmod +x "$SCRIPT_DIR/start.sh"
sudo cp "$SCRIPT_DIR/khedut-voice-pi.service" /etc/systemd/system/khedut-voice-pi.service
sudo sed -i "s|/home/pi/hasmukh/khedut-voice-ai|$SCRIPT_DIR|g" \
    /etc/systemd/system/khedut-voice-pi.service
sudo systemctl daemon-reload
sudo systemctl enable khedut-voice-pi
echo "      ✅ Service installed and enabled (will auto-start on next boot)."
echo ""
echo "  ─────────────────────────────────────────────"
echo "  Useful commands:"
echo "    Start now  :  sudo systemctl start khedut-voice-pi"
echo "    Stop       :  sudo systemctl stop khedut-voice-pi"
echo "    Restart    :  sudo systemctl restart khedut-voice-pi"
echo "    Status     :  sudo systemctl status khedut-voice-pi"
echo "    Live logs  :  journalctl -u khedut-voice-pi -f"
echo "  ─────────────────────────────────────────────"
echo ""

read -p "  Start the Voice AI right now? [y/N]: " START_NOW
if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
    sudo systemctl start khedut-voice-pi
    echo "  Started! Follow logs with:"
    echo "    journalctl -u khedut-voice-pi -f"
else
    echo "  Not started. Run 'sudo systemctl start khedut-voice-pi' when ready."
fi
echo ""
