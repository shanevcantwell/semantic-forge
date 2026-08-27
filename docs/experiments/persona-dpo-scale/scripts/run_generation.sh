#!/bin/bash
# Scale production orchestrator for persona-dpo-scale run0.
# Runs gen_pairs.py once per scenario (one process per call — issue #7 workaround).
# Collects successful rows into pairs.jsonl, logs all attempts to attempts.log.

set -euo pipefail

RUN_DIR="docs/experiments/persona-dpo-scale/run0"
STIMULI="$RUN_DIR/scenarios/bramble/stimuli.txt"
SCRIPTS="docs/experiments/persona-dpo-scale/scripts"

# Persona system prompts — read from card YAMLs directly via python3 -c to avoid yq dependency
BBRAMBLE_SYS=$(python3 -c "import yaml; print(yaml.safe_load(open('docs/experiments/persona-dpo-probe/cards/bramble.yaml'))['system_prompt'])")
VEX_SYS=$(python3 -c "import yaml; print(yaml.safe_load(open('docs/experiments/persona-dpo-probe/cards/vex.yaml'))['system_prompt'])")
MARIGOLD_SYS=$(python3 -c "import yaml; print(yaml.safe_load(open('docs/experiments/persona-dpo-probe/cards/marigold.yaml'))['system_prompt'])")

mkdir -p "$RUN_DIR/scenarios/bramble/pairs" \
         "$RUN_DIR/scenarios/vex/pairs" \
         "$RUN_DIR/scenarios/marigold/pairs"

LOG="$RUN_DIR/generation.log"
echo "=== persona-dpo-scale run0 generation started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"

# Read scenarios into array (40 lines, one scenario per line)
mapfile -t SCENARIOS < <(grep -v '^$' "$STIMULI")
N_SCEN=${#SCENARIOS[@]}
echo "[run-gen] $N_SCEN unique stimuli loaded from $STIMULI" | tee -a "$LOG"

for persona in bramble vex marigold; do
    case $persona in
        bramble) SYS_PROMPT="$BBRAMBLE_SYS" ;;
        vex)     SYS_PROMPT="$VEX_SYS" ;;
        marigold) SYS_PROMPT="$MARIGOLD_SYS" ;;
    esac

    OUTDIR="$RUN_DIR/scenarios/$persona/pairs"
    PAIRS_FILE="$OUTDIR/pairs.jsonl"
    : > "$PAIRS_FILE"  # truncate

    echo "[run-gen] === Generating $persona pairs ($N_SCEN scenarios) ===" | tee -a "$LOG"

    for i in $(seq 0 $((N_SCEN - 1))); do
        scenario="${SCENARIOS[$i]}"
        out_file="$OUTDIR/${i}.json"

        echo "[run-gen] $persona idx=$i starting..." >> "$LOG"

        # One process per invocation — dodges issue #7 SK teardown bug (D-003)
        if python3 "$SCRIPTS/gen_pairs.py" \
            --persona "$persona" \
            --system-prompt "$SYS_PROMPT" \
            --scenario "$scenario" \
            --idx "$i" \
            --out "$out_file" 2>>"$LOG"; then

            # Check if the output row is valid (not an error)
            if python3 -c "import json; d=json.load(open('$out_file')); exit(0 if not d.get('_isError') else 1)" 2>/dev/null; then
                cat "$out_file" >> "$PAIRS_FILE"
                echo "[run-gen] $persona idx=$i ✓ appended to pairs.jsonl" | tee -a "$LOG"
            else
                echo "[run-gen] $persona idx=$i ⚠ error row (skipped from dataset)" | tee -a "$LOG"
            fi
        else
            echo "[run-gen] $persona idx=$i ✗ process failed" | tee -a "$LOG"
        fi

        # Brief pause to avoid hammering the local server
        sleep 0.5
    done

    ROWS=$(wc -l < "$PAIRS_FILE")
    echo "[run-gen] === $persona complete: $ROWS valid pairs ===" | tee -a "$LOG"
done

echo "=== generation finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
