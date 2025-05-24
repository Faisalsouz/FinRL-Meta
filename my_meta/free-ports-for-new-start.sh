#!/bin/bash

PORTS=(3000 8000)  # Ports to check
KILLED=false       # Flag to track if any process was killed

for port in "${PORTS[@]}"; do
  # Find the Process ID (PID) using the port
  pid=$(lsof -ti :"$port" | head -n 1)

  if [[ -n "$pid" ]]; then
    echo "→ Killing process $pid running on port $port..."
    kill -9 "$pid"
    KILLED=true
  else
    echo "→ No process found on port $port."
  fi
done

if [[ "$KILLED" == true ]]; then
  echo "✔ Done: Processes killed successfully."
else
  echo "✔ No processes were running on ports 3000 or 8000."
fi