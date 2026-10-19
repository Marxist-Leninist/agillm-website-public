#!/bin/bash
# keeps serve64b.py up (restarts within ~20 s after a crash); claude-chat-64b-improve-20261019
while true; do pgrep -f "serve64b[.]py" > /dev/null || setsid nohup bash /root/claude64b_cpuinfer/run_serve.sh > /dev/null 2>&1 < /dev/null & sleep 20; done
