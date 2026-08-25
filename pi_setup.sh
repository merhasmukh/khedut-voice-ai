#!/bin/bash
# =============================================================================
# Khedut Voice AI -- Raspberry Pi Setup Script (Python 3.13)
# =============================================================================
# Installs system audio packages, sets up Python 3.13 environment, and installs
# lightweight Voice AI dependencies on Raspberry Pi.
#
# Usage:  chmod +x pi_setup.sh && ./pi_setup.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_TARGET="3.13"
PYTHON_BUILD_VER="3.13.2"

echo "=============================================="
echo "  Khedut Voice AI -- Raspberry Pi Setup 🍓"
echo "  Target Python: Python ${PYTHON_TARGET}"
echo "=============================================="
echo ""

# -- 1. System packages -------------------------------------------------------
echo "[1/5] Installing system audio & build packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    wget \
    curl \
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

# -- 2. Check or install Python 3.13 -------------------------------------------
echo "[2/5] Checking for Python ${PYTHON_TARGET}..."
if command -v python3.13 >/dev/null 2>&1; then
    PY_BIN=$(command -v python3.13)
    echo "      Found Python 3.13 at: ${PY_BIN}"
    ${PY_BIN} --version
else
    echo "      Python 3.13 not found on system."
    echo "      Building & installing Python ${PYTHON_BUILD_VER} from source (this takes ~5-10 mins)..."
    TMP_BUILD_DIR="/tmp/python313_build"
    mkdir -p "${TMP_BUILD_DIR}"
    cd "${TMP_BUILD_DIR}"
    
    wget -q --show-progress "https://www.python.org/ftp/python/${PYTHON_BUILD_VER}/Python-${PYTHON_BUILD_VER}.tgz"
    tar -xf "Python-${PYTHON_BUILD_VER}.tgz"
    cd "Python-${PYTHON_BUILD_VER}"
    
    ./configure --enable-optimizations --prefix=/usr/local
    make -j"$(nproc)"
    sudo make altinstall
    
    cd "$SCRIPT_DIR"
    rm -rf "${TMP_BUILD_DIR}"
    
    PY_BIN="/usr/local/bin/python3.13"
    echo "      Python ${PYTHON_BUILD_VER} successfully installed at ${PY_BIN}."
fi
echo ""

# -- 3. Bluetooth speaker guide -----------------------------------------------
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

# -- 4. Python 3.13 virtual environment ---------------------------------------
echo "[3/5] Setting up Python 3.13 virtual environment..."
# If existing venv is not Python 3.13, recreate it
if [ -d "venv" ]; then
    CURRENT_VENV_PY=$(venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
    if [ "${CURRENT_VENV_PY}" != "${PYTHON_TARGET}" ]; then
        echo "      Existing venv was Python ${CURRENT_VENV_PY}. Recreating with Python ${PYTHON_TARGET}..."
        rm -rf venv
        ${PY_BIN} -m venv venv
    else
        echo "      Found existing Python 3.13 venv."
    fi
else
    ${PY_BIN} -m venv venv
    echo "      Created new Python 3.13 venv."
fi

source venv/bin/activate
pip install --upgrade pip --quiet
echo "      Active venv version: $(python --version)"
echo ""

# -- 5. Install Pi dependencies -----------------------------------------------
echo "[4/5] Installing lightweight Voice AI dependencies (requirements-pi.txt)..."
pip install -r requirements-pi.txt --quiet
echo "      Dependencies installed."
echo ""

# -- 6. Verify .env -----------------------------------------------------------
echo "[5/5] Checking .env configuration..."
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
echo "  Setup Complete with Python 3.13! 🚀"
echo "=============================================="
echo ""
echo "  Start the always-listening Voice AI:"
echo "    source venv/bin/activate"
echo "    python pi_voice_agent.py"
echo ""
echo "  To auto-start on boot (systemd):"
echo "    sudo cp khedut-voice-pi.service /etc/systemd/system/"
echo "    sudo sed -i \"s|/home/pi/khedut-voice-ai|$SCRIPT_DIR|g\" /etc/systemd/system/khedut-voice-pi.service"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable khedut-voice-pi"
echo "    sudo systemctl start khedut-voice-pi"
echo ""
