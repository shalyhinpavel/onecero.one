#!/bin/bash
# OneCeroOne Auto-Update Script

echo "🚀 Starting 1C1 update..."

# 1. Pull latest code
git pull origin main

# 2. Update Python Sidecar dependencies
if [ -d ".venv" ]; then
    source .venv/bin/activate
    pip install -r sidecar/requirements.txt
fi

# 3. Build Rust Core
cd core
source $HOME/.cargo/env
cargo build --release

# 4. Restarting service 
# Note: In a real production setup, this would be handled by systemd or docker-compose.
# For now, we assume the user/owner will restart or the script can try to start it detached.
echo "✅ Update complete. Restarting system..."
nohup ./target/release/oneceroone-core > output.log 2>&1 &
