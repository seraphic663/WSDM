#!/usr/bin/env python3
"""Self-extracting experiment installer.

Run on server:
    python3 /root/data/LRAT/solution/experiments/install.py

This single file contains all experiment scripts embedded as data.
It extracts them to /root/data/LRAT/solution/experiments/.
"""

import os
import sys
from pathlib import Path

BASE = Path("/root/data/LRAT/solution/experiments")
FILES = {}  # Will be populated below

# ============================================================
# Embedded file contents
# ============================================================

FILES["common.py"] = r'''
#!/usr/bin/env python3
"""LRAT Experiment Shared Utilities."""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

SERVER_ROOT = Path("/root/data/LRAT")
CCIR = SERVER_ROOT / "ccir"
DATA_RAW = CCIR / "data" / "raw"
DATA_EXP = CCIR / "data" / "experiments"
EARLY_STOP_V1 = DATA_EXP / "early_stop_v1"
OUTPUTS = CCIR / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
EXPERIMENTS = OUTPUTS / "experiments"
MODELS = CCIR / "models"
VENV_PYTHON = SERVER_ROOT / ".venv" / "bin" / "python"

M00_PATH = MODELS / "Qwen3-Embedding-0.6B"
M01_PATH = MODELS / "Qwen3-Embedding-0.6B-LRAT-1epoch-20260716"
TRAIN_94K = EARLY_STOP_V1 / "train.jsonl"
DEV_1500 = EARLY_STOP_V1 / "dev.jsonl"
TRAJECTORY_TAR = DATA_RAW / "LRAT-trajectories.tar.gz"
PROVENANCE_PATH = EXPERIMENTS / "trajectory_quality_v1_20260724" / "trajectory_provenance_v1.jsonl"

M00_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
M01_SHA = "b25b3b08a3199a788ea5e8bc005ee20ab831ada36cb4eccd7ba915e0d6e02501"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def mark_completed(output_dir, completed_at=None):
    from datetime import datetime
    if completed_at is None:
        completed_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
    with open(Path(output_dir) / "COMPLETED", "w") as f:
        f.write(f"completed_at={completed_at}\n")

def mark_failed(output_dir, reason):
    with open(Path(output_dir) / "FAILED", "w") as f:
        f.write(f"reason={reason}\n")
'''

FILES["m08_trajectory_weighting/prepare.py"] = r'''
#!/usr/bin/env python3
"""M08: Trajectory-level success/failure weighting.

Model: M08
Source: M00 (Qwen3-Embedding-0.6B)
Data: early_stop_v1 train 94,113 rows
Method: Success trajectory reweight_rate * 1.2, Failed * 0.8
"""

import json, sys, tarfile
from pathlib import Path
from collections import Counter
sys.path.insert(0, "/root/data/LRAT/solution/experiments")
from common import *

OUTPUT_DIR = Path("/root/data/LRAT/ccir/data/experiments/m08_trajectory_weighting")
OUTPUT_TRAIN = OUTPUT_DIR / "train_m08.jsonl"

def load_provenance_mapping():
    mapping = {}
    if not PROVENANCE_PATH.exists():
        print(f"ERROR: {PROVENANCE_PATH} not found")
        return mapping
    for row in load_jsonl(PROVENANCE_PATH):
        key = (row.get("query_normalized", row.get("query", "")),
               row.get("positive_doc_id", ""))
        tid = row.get("trajectory_id", "")
        if key[0] and tid:
            mapping[key] = tid
    return mapping

def load_trajectory_status():
    status = {}
    extracted_dir = DATA_RAW / "LRAT-trajectories"
    def collect(rows):
        for row in rows:
            tid = row.get("trajectory_id") or row.get("task_id") or row.get("id")
            fs = row.get("final_status") or row.get("status") or ""
            if tid: status[tid] = fs.lower()
    if extracted_dir.exists():
        for f in sorted(extracted_dir.glob("*.jsonl")):
            collect(load_jsonl(f))
    elif TRAJECTORY_TAR.exists():
        with tarfile.open(TRAJECTORY_TAR, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        rows = [json.loads(l) for l in f.read().decode("utf-8").strip().split("\n") if l.strip()]
                        collect(rows)
    return status

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("M08: Loading provenance...")
    prov_map = load_provenance_mapping()
    print(f"  {len(prov_map)} entries")
    print("M08: Loading trajectory statuses...")
    traj_status = load_trajectory_status()
    successes = sum(1 for v in traj_status.values() if v in ("success","completed","true","1"))
    failures = sum(1 for v in traj_status.values() if v in ("failed","failure","false","0","error"))
    print(f"  Success: {successes}, Failed: {failures}")
    print("M08: Loading training data...")
    train_rows = load_jsonl(TRAIN_94K)
    modified = 0
    for row in train_rows:
        query = row.get("query", "")
        query_norm = " ".join(query.strip().lower().split())
        pos_docs = row.get("positive_passages") or row.get("positive") or []
        if isinstance(pos_docs, dict): pos_docs = [pos_docs]
        if not pos_docs: continue
        found = False
        for pos in pos_docs:
            doc_id = pos.get("doc_id") or pos.get("docid") or pos.get("id", "")
            tid = prov_map.get((query_norm, str(doc_id))) or prov_map.get((query, str(doc_id)))
            if tid and tid in traj_status:
                status = traj_status[tid]
                orig = row.get("reweight_rate", 1.0)
                if status in ("success","completed","true","1"):
                    row["reweight_rate"] = orig * 1.2
                else:
                    row["reweight_rate"] = orig * 0.8
                modified += 1; found = True; break
        if not found: row["reweight_rate"] = row.get("reweight_rate", 1.0)
    print(f"  Modified: {modified}/{len(train_rows)}")
    save_jsonl(OUTPUT_TRAIN, train_rows)
    print(f"  Saved: {OUTPUT_TRAIN}")
    print(f"  SHA: {sha256_file(OUTPUT_TRAIN)}")

if __name__ == "__main__":
    main()
'''

FILES["m10_lr_temp_scan/run_scan.sh"] = r'''#!/bin/bash
# M10: LR + Temperature Scan Runner
set -euo pipefail
cd /root/data/LRAT
source .venv/bin/activate
TRAIN_DATA="ccir/data/experiments/early_stop_v1/train.jsonl"
DEV_DATA="ccir/data/experiments/early_stop_v1/dev.jsonl"
SCAN_DIR="ccir/outputs/experiments/m10_lr_temp_scan"
RESULTS="${SCAN_DIR}/scan_results.jsonl"
mkdir -p "${SCAN_DIR}"
echo "" > "${RESULTS}"

run_one() {
    local name="$1" lr="$2" temp="$3" steps="${4:-500}"
    echo "=== ${name}: LR=${lr} temp=${temp} ==="
    local out="ccir/outputs/experiments/${name}/checkpoints"
    local eval_out="${SCAN_DIR}/eval_${name}.json"
    bash solution/experiments/train.sh "${name}" "${TRAIN_DATA}" --steps "${steps}" --lr "${lr}" --temp "${temp}"
    .venv/bin/python solution/src/evaluate_qwen3_pairs.py --model_path "${out}" --eval_data "${DEV_DATA}" --output_path "${eval_out}" 2>/dev/null || true
    .venv/bin/python -c "
import json
try:
    d=json.load(open('${eval_out}'))
    r={'name':'${name}','lr':'${lr}','temp':'${temp}','R1':d.get('recall@1',0),'MRR':d.get('mrr',0)}
    json.dump(r,open('${RESULTS}','a'))
    open('${RESULTS}','a').write('\n')
    print(f'  R@1={r[\"R1\"]:.4f} MRR={r[\"MRR\"]:.6f}')
except: print('  eval failed')
"
}

echo "=== Phase 1: LR Scan ==="
for lr in 3e-7 5e-7 1e-6 2e-6 3e-6; do
    run_one "m10_phase1_lr${lr}" "${lr}" "0.02" 500
done

echo "=== Phase 2: Temp Scan ==="
BEST_LR=$(python -c "
import json; results=[]
with open('${RESULTS}') as f:
    for l in f:
        l=l.strip()
        if l and 'phase1' in l: results.append(json.loads(l))
print(max(results,key=lambda r:r['MRR'])['lr'] if results else '1e-6')
")
echo "Best LR: ${BEST_LR}"
for temp in 0.01 0.02 0.03 0.05; do
    run_one "m10_phase2_temp${temp}" "${BEST_LR}" "${temp}" 500
done

echo "=== Phase 3: Full Epoch ==="
BEST_TEMP=$(python -c "
import json; results=[]
with open('${RESULTS}') as f:
    for l in f:
        l=l.strip()
        if l and 'phase2' in l: results.append(json.loads(l))
if not results:
    with open('${RESULTS}') as f:
        for l in f:
            l=l.strip()
            if l and 'phase1' in l: results.append(json.loads(l))
print(max(results,key=lambda r:r['MRR'])['temp'] if results else '0.02')
")
echo "Best combo: LR=${BEST_LR} temp=${BEST_TEMP}"
bash solution/experiments/train.sh "m10_best_full" "${TRAIN_DATA}" --lr "${BEST_LR}" --temp "${BEST_TEMP}"
echo "M10 complete."
'''

FILES["m12_sequence_weight/prepare.py"] = r'''
#!/usr/bin/env python3
"""M12: Sequence-Level Turn Weighting.
Model: M12  Source: M00
Method: Later turns get higher reweight_rate (×1.0→1.1→1.2)
"""
import json, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, "/root/data/LRAT/solution/experiments")
from common import *

OUTPUT_DIR = Path("/root/data/LRAT/ccir/data/experiments/m12_sequence_weight")
OUTPUT_TRAIN = OUTPUT_DIR / "train_m12.jsonl"

def load_provenance_with_turns():
    mapping = {}
    if not PROVENANCE_PATH.exists():
        return mapping
    for row in load_jsonl(PROVENANCE_PATH):
        key = (row.get("query_normalized", row.get("query", "")),
               row.get("positive_doc_id", ""))
        turn = row.get("turn_number") or row.get("turn") or 1
        try: turn = int(turn)
        except: turn = 1
        if key[0]: mapping[key] = turn
    return mapping

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prov_map = load_provenance_with_turns()
    print(f"M12: {len(prov_map)} provenance entries")
    train_rows = load_jsonl(TRAIN_94K)
    modified = 0
    for row in train_rows:
        query = row.get("query", "")
        query_norm = " ".join(query.strip().lower().split())
        pos_docs = row.get("positive_passages") or row.get("positive") or []
        if isinstance(pos_docs, dict): pos_docs = [pos_docs]
        if not pos_docs: continue
        found = False
        for pos in pos_docs:
            doc_id = pos.get("doc_id") or pos.get("docid") or pos.get("id", "")
            turn = prov_map.get((query_norm, str(doc_id))) or prov_map.get((query, str(doc_id)))
            if turn:
                w = 1.0 if turn == 1 else (1.1 if turn <= 3 else 1.2)
                row["reweight_rate"] = row.get("reweight_rate", 1.0) * w
                modified += 1; found = True; break
        if not found: row["reweight_rate"] = row.get("reweight_rate", 1.0)
    print(f"M12: Modified {modified}/{len(train_rows)}")
    save_jsonl(OUTPUT_TRAIN, train_rows)
    print(f"M12: Saved {OUTPUT_TRAIN} SHA={sha256_file(OUTPUT_TRAIN)}")

if __name__ == "__main__":
    main()
'''

FILES["train.sh"] = r'''#!/bin/bash
# LRAT Experiment Training Wrapper
set -euo pipefail
EXPERIMENT_ID="${1:?}"
TRAIN_DATA="${2:?}"
shift 2 || true
cd /root/data/LRAT
source .venv/bin/activate
MODEL_PATH="ccir/models/Qwen3-Embedding-0.6B"
CHECKPOINT_DIR="ccir/outputs/experiments/${EXPERIMENT_ID}/checkpoints"
CACHE_DIR="ccir/data/cache/${EXPERIMENT_ID}"
LOG_DIR="ccir/outputs/logs"
mkdir -p "${CHECKPOINT_DIR}" "${CACHE_DIR}" "${LOG_DIR}"
MAX_STEPS=""
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --short) MAX_STEPS="--max_steps 500"; shift ;;
        --steps) MAX_STEPS="--max_steps $2"; shift 2 ;;
        --lr) EXTRA_ARGS="${EXTRA_ARGS} --learning_rate $2"; shift 2 ;;
        --temp) EXTRA_ARGS="${EXTRA_ARGS} --temperature $2"; shift 2 ;;
        *) EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${EXPERIMENT_ID}_${TIMESTAMP}.log"
echo "=== ${EXPERIMENT_ID} ==="
echo "Train: ${TRAIN_DATA} Output: ${CHECKPOINT_DIR}"
if [ -f "${CHECKPOINT_DIR}/COMPLETED" ]; then
    echo "Already completed. Skip."
    exit 0
fi
echo "running_at=$(date -Is)" > "${CHECKPOINT_DIR}/RUNNING"
set +e
torchrun --standalone --nproc_per_node=2 \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    --model_name_or_path "${MODEL_PATH}" \
    --train_data "${TRAIN_DATA}" \
    --cache_path "${CACHE_DIR}" \
    --train_group_size 6 \
    --query_max_len 128 --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval "Given a web search query, retrieve relevant passages that answer the query" \
    --query_instruction_format $'Instruct: {}\nQuery:{}' \
    --knowledge_distillation False \
    --output_dir "${CHECKPOINT_DIR}" \
    --learning_rate 1e-6 --bf16 --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --dataloader_drop_last True --warmup_ratio 0.1 \
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 10 --save_strategy steps --save_steps 500 --save_total_limit 3 \
    --negatives_cross_device --temperature 0.02 \
    --sentence_pooling_method last_token --normalize_embeddings True \
    --seed 20260716 --data_seed 20260716 \
    --ddp_find_unused_parameters False --report_to none \
    ${MAX_STEPS} ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"
RET=$?
rm -f "${CHECKPOINT_DIR}/RUNNING"
if [ ${RET} -eq 0 ]; then
    echo "completed_at=$(date -Is)" > "${CHECKPOINT_DIR}/COMPLETED"
    echo "=== COMPLETED ==="
else
    echo "reason=exit_${RET}" > "${CHECKPOINT_DIR}/FAILED"
    echo "=== FAILED (${RET}) ==="
fi
exit ${RET}
'''

FILES["run_all.sh"] = r'''#!/bin/bash
# M08-M13 Experiment Runner
set -euo pipefail
cd /root/data/LRAT
source .venv/bin/activate
echo "=== M08-M13 Experiment Suite ==="
echo "Started: $(date -Is)"

for exp in m08_trajectory_weighting m12_sequence_weight; do
    echo "===== ${exp} ====="
    id="${exp}"
    data="ccir/data/experiments/${exp}/train_${exp#m[0-9][0-9]_}.jsonl"
    [ "$exp" = "m08_trajectory_weighting" ] && data="ccir/data/experiments/m08_trajectory_weighting/train_m08.jsonl"
    [ "$exp" = "m12_sequence_weight" ] && data="ccir/data/experiments/m12_sequence_weight/train_m12.jsonl"
    python solution/experiments/${exp}/prepare.py
    bash solution/experiments/train.sh "${exp}_500" "${data}" --short
done

echo "=== M10 LR/Temp Scan ==="
bash solution/experiments/m10_lr_temp_scan/run_scan.sh

echo "=== Suite Complete: $(date -Is) ==="
'''


# ============================================================
# Extraction logic
# ============================================================

def extract_all():
    """Extract all embedded files to the server."""
    BASE.mkdir(parents=True, exist_ok=True)

    for rel_path, content in FILES.items():
        target = BASE / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        # Strip leading/trailing whitespace from content
        content = content.strip()

        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")

        # Make shell scripts executable
        if rel_path.endswith(".sh"):
            os.chmod(target, 0o755)

        print(f"  {rel_path} ({len(content)} bytes)")

    print(f"\nExtracted {len(FILES)} files to {BASE}")

def run_prepare_scripts():
    """Run data preparation for experiments that don't need GPU."""
    import subprocess

    scripts = [
        ("M08", "m08_trajectory_weighting/prepare.py"),
        ("M12", "m12_sequence_weight/prepare.py"),
    ]

    for label, script in scripts:
        script_path = BASE / script
        if script_path.exists():
            print(f"\n[{label}] Preparing data...")
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True, text=True, timeout=300
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[{label}] ERROR: {result.stderr}")
        else:
            print(f"[{label}] Script not found: {script_path}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
        extract_all()
        return

    extract_all()

    if "--run" in sys.argv:
        run_prepare_scripts()

        # Start M10 scan
        print("\n=== Starting M10 LR/Temp Scan ===")
        import subprocess
        subprocess.run(["bash", str(BASE / "m10_lr_temp_scan/run_scan.sh")])

if __name__ == "__main__":
    main()
