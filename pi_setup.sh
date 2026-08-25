#!/bin/bash
# =============================================================================
# Khedut Voice AI -- Raspberry Pi Setup Script
# =============================================================================
# Run this ONCE on your Raspberry Pi to install all dependencies.
# Usage:  chmod +x pi_setup.sh && ./pi_setup.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Khedut Voice AI -- Raspberry Pi Setup"
echo "=============================================="
echo ""

# -- 1. System packages -------------------------------------------------------
echo "[1/5] Installing system packages..."
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

# -- 2. Bluetooth speaker help ------------------------------------------------
echo "=============================================="
echo "  Bluetooth Speaker Pairing Guide"
echo "=============================================="
echo "  Run these commands to pair your Bluetooth speaker:"
echo "    bluetoothctl"
echo "    > power on"
echo "    > scan on"
echo "    > pair   XX:XX:XX:XX:XX:XX   (replace with your speaker MAC)"
echo "    > connect XX:XX:XX:XX:XX:XX"
echo "    > trust  XX:XX:XX:XX:XX:XX"
echo "    > exit"
echo ""
echo "  Then set it as the default audio output:"
echo "    pactl list sinks short"
echo "    pactl set-default-sink bluez_sink.XX_XX_XX_XX_XX_XX.a2dp_sink"
echo ""
echo "  Press Enter to continue setup..."
read -r

# -- 3. Python venv -----------------------------------------------------------
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "      Created new venv."
else
    echo "      Found existing venv."
fi
source venv/bin/activate
pip install --upgrade pip --quiet
echo "      venv ready."
echo ""

# -- 4. Python dependencies ---------------------------------------------------
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "      Python packages installed."
echo ""

# -- 5. Verify .env -----------------------------------------------------------
echo "[4/5] Checking .env configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "      Created .env from .env.example"
        echo "      *** Please edit .env and add your GEMINI_API_KEY ***"
    else
        echo "      WARNING: No .env file found."
        echo "      Create one with:  echo 'GEMINI_API_KEY=your_key_here' > .env"
    fi
else
    if grep -q "GEMINI_API_KEY=" .env; then
        echo "      .env found with GEMINI_API_KEY."
    else
        echo "      WARNING: GEMINI_API_KEY not found in .env"
    fi
fi
echo ""

# -- 6. List audio devices ----------------------------------------------------
echo "[5/5] Available audio devices:"
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
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Start the Pi voice agent:"
echo "    source venv/bin/activate"
echo "    python pi_voice_agent.py"
echo ""
echo "  Use specific audio devices if needed:"
echo "    python pi_voice_agent.py --list-devices"
echo "    python pi_voice_agent.py --input-device 2 --output-device 5"
echo ""
echo "  To auto-start on boot (systemd):"
echo "    sudo cp khedut-voice-pi.service /etc/systemd/system/"
echo "    sudo sed -i \"s|/home/pi/khedut-voice-ai|$SCRIPT_DIR|g\" /etc/systemd/system/khedut-voice-pi.service"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable khedut-voice-pi"
echo "    sudo systemctl start khedut-voice-pi"
echo ""
echo "  Run web/browser mode (unchanged):"
echo "    uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
