#!/bin/bash

PORTS=(3000 8000)  # Ports to check
KILLED=false       # Flag to track if any process was killed
MAX_ATTEMPTS=3     # Maximum attempts to kill processes
ATTEMPT_DELAY=1    # Delay between attempts in seconds

# Function to check if port is actually free
is_port_free() {
  local port=$1
  ! lsof -i :"$port" >/dev/null 2>&1 && ! ss -tuln | grep -q ":$port " && ! netstat -tuln | grep -q ":$port "
}

for port in "${PORTS[@]}"; do
  ATTEMPT=1
  while [[ $ATTEMPT -le $MAX_ATTEMPTS ]]; do
    # Find all PIDs using the port EXCEPT ngrok
    pids=()
    while IFS= read -r pid; do
      # Skip ngrok processes
      if ! ps -p "$pid" -o command= | grep -q "ngrok"; then
        pids+=("$pid")
      fi
    done < <(lsof -ti :"$port" | sort -u)
    
    if [[ ${#pids[@]} -gt 0 ]]; then
      echo "→ Found ${#pids[@]} process(es) (excluding ngrok) running on port $port:"
      lsof -i :"$port" | grep -v "ngrok" || true
      
      echo "→ Killing process(es): ${pids[*]}..."
      kill -9 "${pids[@]}" 2>/dev/null || true
      KILLED=true
      
      sleep $ATTEMPT_DELAY
    else
      echo "→ No non-ngrok processes found on port $port."
      break
    fi
    
    ((ATTEMPT++))
  done
  
  # Final check for port status
  if is_port_free "$port"; then
    echo "✓ Port $port is now free (ngrok may still be running)."
  else
    echo "⚠ Port $port still in use (possibly by ngrok)."
  fi
done

if [[ "$KILLED" == true ]]; then
  echo "✔ Done: Non-ngrok processes killed successfully."
else
  echo "✔ No non-ngrok processes were running on the specified ports."
fi