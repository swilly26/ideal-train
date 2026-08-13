#!/bin/bash
export ALPACA_API_KEY="PKSM2QGK2KQUUALPZ4CM56F7F3"
export ALPACA_SECRET_KEY="7d8tkoJ9Xw5X8XfJgmhkxcjpWqCN5u6AHUdAXErsWS8D"
export ALPACA_PAPER="true"
cd /home/team/shared/engine
source .venv/bin/activate
timeout 120 python3 quick_test.py 2>&1
