#!/bin/bash

# ===== USER CONFIGURATION =====
FILE_PATHS=(
    "/home/souz_wsl/finrl_proj/FinRL_Meta/my_meta"
    "/home/souz_wsl/finrl_proj/FinRL_Meta/frontend_crp/frontend-crp/app"
    "/home/souz_wsl/finrl_proj/FinRL_Meta/api"
)

ALLOWED_EXTENSIONS=("py" "log" "tsx" "md" "yml" "css" "ts")  # Add more if needed

BACKUP_DIR="$HOME/text-backup-crypto-api"
mkdir -p "$BACKUP_DIR"
OUTPUT_FILE="$BACKUP_DIR/repo_summary_$(date +'%Y-%m-%d_%H-%M-%S').md"

echo "Creating Markdown summary at: $OUTPUT_FILE"
echo -e "# 🧠 Combined Source File Summary\n\n" > "$OUTPUT_FILE"

# ===== FUNCTION TO HANDLE A DIRECTORY =====
process_directory() {
    local DIR="$1"
    local BASE_DIR=$(realpath "$DIR")

    find "$DIR" -type f | while read -r file; do
        file_ext="${file##*.}"
        file_ext_lower=$(echo "$file_ext" | tr '[:upper:]' '[:lower:]')

        if [[ " ${ALLOWED_EXTENSIONS[*]} " =~ " $file_ext_lower " ]]; then
            rel_path=$(realpath --relative-to="$BASE_DIR" "$file")
            echo -e "## 📄 \`$DIR/$rel_path\`\n" >> "$OUTPUT_FILE"
            echo -e '```' >> "$OUTPUT_FILE"
            cat "$file" >> "$OUTPUT_FILE"
            echo -e '\n```\n' >> "$OUTPUT_FILE"
            echo "✔ Processed: $file"
        else
            echo "✘ Skipped (extension): $file"
        fi
    done
}

# ===== LOOP THROUGH PROVIDED PATHS =====
for file_path in "${FILE_PATHS[@]}"; do
    if [[ -f "$file_path" ]]; then
        echo -e "## 📄 \`$file_path\`\n" >> "$OUTPUT_FILE"
        echo -e '```' >> "$OUTPUT_FILE"
        cat "$file_path" >> "$OUTPUT_FILE"
        echo -e '\n```\n' >> "$OUTPUT_FILE"
        echo "✔ Processed file: $file_path"
    elif [[ -d "$file_path" ]]; then
        process_directory "$file_path"
    else
        echo "⚠ Not found: $file_path"
    fi
done

# ===== APPEND DIRECTORY TREE AT THE END =====
TREE_PATHS=("/home/souz_wsl/finrl_proj/FinRL_Meta")
echo "Appending repo structure map..."
echo -e "\n\n# 🗂 Repo Structure Map\n" >> "$OUTPUT_FILE"
for dir in "${TREE_PATHS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo -e "### 📁 \`$dir\`\n\`\`\`\n" >> "$OUTPUT_FILE"
        tree "$dir" -a -I '.git|node_modules|__pycache__|.next' >> "$OUTPUT_FILE"
        echo -e "\n\`\`\`\n" >> "$OUTPUT_FILE"
    fi
done

echo "✅ Markdown summary created!"
echo "📄 Output saved to: $OUTPUT_FILE"
