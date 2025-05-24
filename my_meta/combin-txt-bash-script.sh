#!/bin/bash

# Fixed file paths to check
FILE_PATHS=(
    "/home/souz_wsl/finrl_proj/FinRL_Meta/my_meta"         
    "/home/souz_wsl/finrl_proj/FinRL_Meta/frontend_crp/frontend-crp/app" 
    "api"      
)

# Allowed file extensions (case-insensitive)
ALLOWED_EXTENSIONS=("py" "log" "tsx" "md" "yml" "css" "ts")  # Add more if needed

# Backup directory (will be created if it doesn't exist)
BACKUP_DIR="$HOME/text-backup-crypto-api"
mkdir -p "$BACKUP_DIR"

# Output filename with current date
OUTPUT_FILE="$BACKUP_DIR/backup_$(date +'%Y-%m-%d_%H-%M-%S').txt"

# Loop through each file path
for file_path in "${FILE_PATHS[@]}"; do
    if [[ -f "$file_path" ]]; then  # Check if file exists
        # Extract file extension (case-insensitive check)
        file_ext="${file_path##*.}"
        file_ext_lower=$(echo "$file_ext" | tr '[:upper:]' '[:lower:]')

        # Check if extension is allowed
        if [[ " ${ALLOWED_EXTENSIONS[*]} " =~ " $file_ext_lower " ]]; then
            # Write filename (with extension) first
            echo "=== FILE: $(basename "$file_path") ===" >> "$OUTPUT_FILE"
            # Write file content
            cat "$file_path" >> "$OUTPUT_FILE"
            # Add 2 newlines for spacing
            echo -e "\n\n" >> "$OUTPUT_FILE"
            echo "Copied: $file_path"
        else
            echo "Skipped (invalid extension): $file_path"
        fi
    else
        echo "Not found: $file_path"
    fi
done

echo "Backup completed! Output: $OUTPUT_FILE"