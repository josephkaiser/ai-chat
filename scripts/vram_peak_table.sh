#!/usr/bin/env bash
set -euo pipefail

interval="${1:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Run this on the NVIDIA host." >&2
    exit 1
fi

if ! [[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Usage: $0 [poll_interval_seconds]" >&2
    exit 1
fi

declare -A gpu_name_by_index
declare -A total_mem_by_index
declare -A peak_mem_by_index
declare -A peak_util_by_index

start_epoch="$(date +%s)"

render_table() {
    local now elapsed
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    elapsed="$(( $(date +%s) - start_epoch ))"

    printf '\033[2J\033[H'
    printf 'VRAM Peak Table  started=%s  elapsed=%ss  interval=%ss\n\n' "$now" "$elapsed" "$interval"
    printf '%-5s %-28s %10s %12s %12s %10s %10s\n' \
        "GPU" "Name" "Total(MB)" "Used(MB)" "Peak(MB)" "Free(MB)" "Util(%)"
    printf '%-5s %-28s %10s %12s %12s %10s %10s\n' \
        "-----" "----------------------------" "---------" "----------" "----------" "--------" "--------"
}

while true; do
    mapfile -t rows < <(
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits
    )

    render_table

    for row in "${rows[@]}"; do
        IFS=',' read -r raw_index raw_name raw_total raw_used raw_free raw_util <<<"$row"

        index="$(echo "$raw_index" | xargs)"
        name="$(echo "$raw_name" | xargs)"
        total="$(echo "$raw_total" | xargs)"
        used="$(echo "$raw_used" | xargs)"
        free="$(echo "$raw_free" | xargs)"
        util="$(echo "$raw_util" | xargs)"

        gpu_name_by_index["$index"]="$name"
        total_mem_by_index["$index"]="$total"

        if [[ -z "${peak_mem_by_index[$index]:-}" ]] || (( used > peak_mem_by_index[$index] )); then
            peak_mem_by_index["$index"]="$used"
        fi
        if [[ -z "${peak_util_by_index[$index]:-}" ]] || (( util > peak_util_by_index[$index] )); then
            peak_util_by_index["$index"]="$util"
        fi

        printf '%-5s %-28.28s %10s %12s %12s %10s %10s\n' \
            "$index" \
            "${gpu_name_by_index[$index]}" \
            "${total_mem_by_index[$index]}" \
            "$used" \
            "${peak_mem_by_index[$index]}" \
            "$free" \
            "$util"
    done

    printf '\nPeak util observed this run:\n'
    for index in "${!peak_util_by_index[@]}"; do
        printf 'GPU %s: %s%% util peak, %s MB VRAM peak\n' \
            "$index" \
            "${peak_util_by_index[$index]}" \
            "${peak_mem_by_index[$index]}"
    done | sort -V

    sleep "$interval"
done
