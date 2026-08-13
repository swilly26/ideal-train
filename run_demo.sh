#!/bin/bash
export ALPACA_API_KEY="PKSM2QGK2KQUUALPZ4CM56F7F3"
export ALPACA_SECRET_KEY="7d8tkoJ9Xw5X8XfJgmhkxcjpWqCN5u6AHUdAXErsWS8D"
export ALPACA_PAPER="true"
cd /home/team/shared/engine
mkdir -p configs
source .venv/bin/activate
timeout 600 python3 full_demo.py 2>&1
