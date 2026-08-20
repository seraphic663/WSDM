#!/usr/bin/env bash
# Research-only reproduction of LRAT's paper data flywheel.
#
# Each loop is deliberately staged:
#   current retriever -> fresh corpus index -> fresh Tongyi trajectories
#   -> Qwen judge + official LRAT data_builder -> 2-epoch weighted InfoNCE
#   -> audited next retriever.
#
# No output from this workflow is competition eligible.
set -euo pipefail

project_root="${PROJECT_ROOT:-/root/data/LRAT}"
config="${PAPER_FLYWHEEL_CONFIG:-${project_root}/solution/configs/paper_flywheel_v1.json}"
research_root="${PAPER_FLYWHEEL_ROOT:-${project_root}/ccir/research/paper_flywheel_v1}"
action="${1:-preflight}"
loop_number="${2:-0}"
current_model="${3:-${project_root}/ccir/models/Qwen3-Embedding-0.6B}"

cd "${project_root}"
python_bin="${project_root}/.venv/bin/python"
loop_tag="$(printf 'loop_%02d' "${loop_number}")"
loop_dir="${research_root}/${loop_tag}"
seed_pool="${research_root}/inputs/InfoSeekQA.pool.tsv"
seed_manifest="${research_root}/inputs/InfoSeekQA.pool.manifest.json"
query_tsv="${loop_dir}/queries.tsv"
query_manifest="${loop_dir}/queries.manifest.json"
source_qa="${research_root}/inputs/InfoSeekQA.source.5b59bfd.jsonl"
corpus="${project_root}/ccir/data/raw/wiki-25-512/wiki-25-512.jsonl"
trajectory_archive="${project_root}/ccir/data/raw/LRAT-trajectories.tar.gz"
agent_model="${PAPER_AGENT_MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
judge_model="${PAPER_JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
agent_port="${PAPER_AGENT_PORT:-6008}"
judge_port="${PAPER_JUDGE_PORT:-6009}"
index_shards="${PAPER_INDEX_SHARDS:-16}"
sampling_seed="${PAPER_QUERY_SAMPLING_SEED:-2025}"

require_research_write() {
  if [[ "${ALLOW_PAPER_FLYWHEEL_RESEARCH:-0}" != "1" ]]; then
    echo "research mutation refused: set ALLOW_PAPER_FLYWHEEL_RESEARCH=1" >&2
    exit 2
  fi
  case "$(realpath -m "${loop_dir}")" in
    "$(realpath "${research_root}")"/*) ;;
    *) echo "loop path escapes research root" >&2; exit 2 ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 2
  fi
}

require_service() {
  local port="$1"
  local expected_model="$2"
  local response
  if ! response="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models")"; then
    echo "required local OpenAI-compatible service is unavailable on port ${port}" >&2
    exit 2
  fi
  if ! "${python_bin}" - "${expected_model}" "${response}" <<'PY'
import json,sys
expected=sys.argv[1]
value=json.loads(sys.argv[2])
ids={str(item.get("id")) for item in value.get("data",[]) if isinstance(item,dict)}
if expected not in ids:
    raise SystemExit(f"service model mismatch: expected {expected!r}, got {sorted(ids)!r}")
PY
  then
    exit 2
  fi
}

require_idle_gpus() {
  if ps -eo args= | grep -E '[v]llm[[:space:]]+serve|[t]orchrun' >/dev/null; then
    echo "index construction requires no vLLM or torchrun process" >&2
    exit 3
  fi
  while IFS=',' read -r gpu memory utilization; do
    memory="${memory//[^0-9]/}"
    utilization="${utilization//[^0-9]/}"
    if (( memory > 2048 || utilization > 20 )); then
      echo "GPU ${gpu} is not idle: ${memory} MiB / ${utilization}%" >&2
      exit 3
    fi
  done < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  free_bytes="$(df -B1 --output=avail /root/data | tail -n 1 | tr -d ' ')"
  if (( free_bytes < 150 * 1024 * 1024 * 1024 )); then
    echo "paper index construction requires at least 150 GiB free" >&2
    exit 3
  fi
}

write_index_manifest() {
  "${python_bin}" - "${loop_dir}" "${current_model}" "${index_shards}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
loop=Path(sys.argv[1]); model=Path(sys.argv[2]); expected=int(sys.argv[3])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()
files=sorted((loop/"index").glob("index-*.pkl"))
if len(files)!=expected:
    raise SystemExit(f"index shard count mismatch: {len(files)} != {expected}")
value={
  "created_at":datetime.now().astimezone().isoformat(),
  "retriever_model":str(model.resolve()),
  "retriever_model_sha256":sha(model/"model.safetensors"),
  "shards":[{"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha(p)} for p in files],
  "complete":True,
}
(loop/"index"/"manifest.json").write_text(json.dumps(value,indent=2)+"\n")
print(json.dumps(value,indent=2))
PY
}

verify_index_manifest() {
  "${python_bin}" - "${loop_dir}/index/manifest.json" "${current_model}" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest=Path(sys.argv[1]); model=Path(sys.argv[2])
value=json.loads(manifest.read_text())
h=hashlib.sha256()
with (model/"model.safetensors").open("rb") as f:
    for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
if value.get("complete") is not True:
    raise SystemExit("existing index manifest is not complete")
if value.get("retriever_model_sha256") != h.hexdigest():
    raise SystemExit("existing index belongs to a different retriever")
for item in value.get("shards", []):
    path=Path(item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise SystemExit(f"missing or changed index shard: {path}")
print("existing index manifest matches the current retriever")
PY
}

finalize_collection() {
  "${python_bin}" - "${query_tsv}" "${loop_dir}/trajectories" <<'PY'
import csv,hashlib,json,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
query_path=Path(sys.argv[1]); trajectory_dir=Path(sys.argv[2])
with query_path.open(encoding="utf-8",newline="") as f:
    expected={row[0] for row in csv.reader(f,delimiter="\t") if row}
seen={}; statuses=Counter(); aggregate=hashlib.sha256()
for path in sorted(trajectory_dir.glob("run_*.json")):
    value=json.loads(path.read_text(encoding="utf-8"))
    query_id=str(value.get("query_id"))
    if query_id in seen: raise SystemExit(f"duplicate query id: {query_id}")
    seen[query_id]=path.name
    statuses[str(value.get("status","unknown"))]+=1
    aggregate.update(path.name.encode()+b"\0")
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): aggregate.update(chunk)
if set(seen)!=expected:
    missing=sorted(expected-set(seen))[:20]
    extra=sorted(set(seen)-expected)[:20]
    raise SystemExit(
        f"trajectory coverage mismatch: {len(seen)}/{len(expected)}, "
        f"missing={missing}, extra={extra}"
    )
value={
  "created_at":datetime.now().astimezone().isoformat(),
  "query_count":len(expected),
  "trajectory_count":len(seen),
  "status_counts":dict(statuses),
  "aggregate_sha256":aggregate.hexdigest(),
  "complete":True,
  "competition_submission_eligible":False,
}
(trajectory_dir/"COLLECTION_MANIFEST.json").write_text(json.dumps(value,indent=2)+"\n")
print(json.dumps(value,indent=2))
PY
}

case "${action}" in
  preflight)
    exec "${python_bin}" solution/src/audit_paper_flywheel_iteration.py \
      --config "${config}" --repo-root "${project_root}"
    ;;

  extract-seeds)
    require_research_write
    mkdir -p "${research_root}/inputs"
    exec "${python_bin}" solution/src/extract_paper_flywheel_seed_queries.py \
      --mode source-pool \
      --archive "${trajectory_archive}" \
      --output-tsv "${seed_pool}" \
      --manifest "${seed_manifest}" \
      --source-qa-jsonl "${source_qa}" \
      --source-revision "5b59bfd201c19e1b537ab1fc06764e00b2a9c627" \
      --expected-count 10000
    ;;

  sample-queries)
    require_research_write
    require_file "${seed_pool}"
    mkdir -p "${loop_dir}"
    exec "${python_bin}" solution/src/extract_paper_flywheel_seed_queries.py \
      --mode sample-loop \
      --pool-tsv "${seed_pool}" \
      --output-tsv "${query_tsv}" \
      --manifest "${query_manifest}" \
      --sample-count 10000 \
      --seed "${sampling_seed}" \
      --loop-number "${loop_number}"
    ;;

  index)
    require_research_write
    require_idle_gpus
    require_file "${current_model}/model.safetensors"
    require_file "${corpus}"
    mkdir -p "${loop_dir}/index" "${loop_dir}/logs"
    if [[ -e "${loop_dir}/index/manifest.json" ]]; then
      verify_index_manifest
      exit 0
    fi
    printf 'started_at=%s\nmodel=%s\n' "$(date -Is)" "${current_model}" > "${loop_dir}/index/RUNNING"
    rm -f "${loop_dir}/index/FAILED"
    status=0
    for ((base=0; base<index_shards; base+=2)); do
      pids=()
      for gpu in 0 1; do
        shard=$((base+gpu))
        if (( shard >= index_shards )); then continue; fi
        output="${loop_dir}/index/index-$(printf '%02d' "${shard}").pkl"
        CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m tevatron.retriever.driver.encode \
          --model_name_or_path "${current_model}" \
          --dataset_name json \
          --dataset_path "${corpus}" \
          --dataset_number_of_shards "${index_shards}" \
          --dataset_shard_index "${shard}" \
          --encode_output_path "${output}" \
          --passage_max_len 512 \
          --normalize \
          --pooling eos \
          --passage_prefix "" \
          --per_device_eval_batch_size 64 \
          --dataloader_num_workers 4 \
          --padding_side left \
          --bf16 \
          > "${loop_dir}/logs/index-${shard}.log" 2>&1 &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then status=1; fi
      done
      if [[ "${status}" -ne 0 ]]; then break; fi
    done
    rm -f "${loop_dir}/index/RUNNING"
    if [[ "${status}" -ne 0 ]]; then
      printf 'failed_at=%s\n' "$(date -Is)" > "${loop_dir}/index/FAILED"
      exit "${status}"
    fi
    write_index_manifest
    printf 'completed_at=%s\n' "$(date -Is)" > "${loop_dir}/index/COMPLETED"
    ;;

  collect)
    require_research_write
    require_file "${query_tsv}"
    require_file "${query_manifest}"
    require_file "${loop_dir}/index/manifest.json"
    require_service "${agent_port}" "${agent_model}"
    mkdir -p "${loop_dir}/trajectories" "${loop_dir}/logs"
    if [[ -e "${loop_dir}/trajectories/COMPLETED" ]]; then
      finalize_collection
      exit 0
    fi
    printf 'started_at=%s\n' "$(date -Is)" \
      > "${loop_dir}/trajectories/RUNNING"
    rm -f "${loop_dir}/trajectories/FAILED"
    # The 30B Agent owns the GPUs. Retrieval and FAISS therefore run on CPU in
    # the client process while still querying the local vLLM Agent service.
    set +e
    MAX_LLM_CALL_PER_RUN=100 CUDA_VISIBLE_DEVICES="" \
      "${python_bin}" search_agent/tongyi_client.py \
      --output-dir "${loop_dir}/trajectories" \
      --searcher-type faiss \
      --index-path "${loop_dir}/index/index-*.pkl" \
      --model-name "${current_model}" \
      --pooling eos \
      --normalize \
      --torch-dtype float32 \
      --dataset-name "${corpus}" \
      --num-threads "${PAPER_AGENT_THREADS:-8}" \
      --model "${agent_model}" \
      --snippet-max-tokens 64 \
      --query "${query_tsv}" \
      --port "${agent_port}" \
      --k 10 \
      2>&1 | tee "${loop_dir}/logs/collect.log"
    status=${PIPESTATUS[0]}
    set -e
    rm -f "${loop_dir}/trajectories/RUNNING"
    if [[ "${status}" -ne 0 ]]; then
      printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" \
        > "${loop_dir}/trajectories/FAILED"
      exit "${status}"
    fi
    if ! finalize_collection; then
      printf 'failed_at=%s\nreason=incomplete_trajectory_coverage\n' \
        "$(date -Is)" > "${loop_dir}/trajectories/FAILED"
      exit 4
    fi
    printf 'completed_at=%s\n' "$(date -Is)" \
      > "${loop_dir}/trajectories/COMPLETED"
    ;;

  build-pairs)
    require_research_write
    require_service "${judge_port}" "${judge_model}"
    require_file "${corpus}"
    require_file "${loop_dir}/trajectories/COMPLETED"
    if [[ ! -d "${loop_dir}/trajectories" ]]; then
      echo "missing trajectory directory: ${loop_dir}/trajectories" >&2
      exit 2
    fi
    if [[ -e "${loop_dir}/training_pairs.jsonl" ]]; then
      echo "training pairs already exist: ${loop_dir}/training_pairs.jsonl" >&2
      exit 2
    fi
    printf 'started_at=%s\n' "$(date -Is)" > "${loop_dir}/PAIRS_RUNNING"
    rm -f "${loop_dir}/PAIRS_FAILED"
    set +e
    "${python_bin}" solution/src/build_paper_flywheel_pairs.py \
      --corpus-path "${corpus}" \
      --traj-dir "${loop_dir}/trajectories" \
      --output-path "${loop_dir}/training_pairs.jsonl" \
      --summary-path "${loop_dir}/training_pairs.summary.json" \
      --tokenizer-path "${current_model}" \
      --judge-api-url "http://127.0.0.1:${judge_port}/v1/chat/completions" \
      --judge-model "${judge_model}" \
      --max-workers "${PAPER_JUDGE_WORKERS:-16}" \
      --future-timeout 600 \
      2>&1 | tee "${loop_dir}/logs/build-pairs.log"
    status=${PIPESTATUS[0]}
    set -e
    rm -f "${loop_dir}/PAIRS_RUNNING"
    if [[ "${status}" -ne 0 ]]; then
      printf 'failed_at=%s\nexit_code=%s\n' "$(date -Is)" "${status}" \
        > "${loop_dir}/PAIRS_FAILED"
      exit "${status}"
    fi
    printf 'completed_at=%s\n' "$(date -Is)" > "${loop_dir}/PAIRS_COMPLETED"
    ;;

  train)
    require_research_write
    require_file "${loop_dir}/training_pairs.jsonl"
    require_file "${loop_dir}/PAIRS_COMPLETED"
    exec bash solution/scripts/train_paper_flywheel_retriever.sh \
      "${current_model}" \
      "${loop_dir}/training_pairs.jsonl" \
      "${loop_dir}/next_retriever" \
      "${loop_dir}/logs/train.log"
    ;;

  audit)
    exec "${python_bin}" solution/src/audit_paper_flywheel_iteration.py \
      --config "${config}" \
      --repo-root "${project_root}" \
      --loop-dir "${loop_dir}" \
      --current-model "${current_model}" \
      --require-completed-training
    ;;

  plan)
    cat <<EOF
Paper flywheel ${loop_tag} (research only; never competition eligible)
1. Deterministically sample this loop's 10K queries from the full InfoSeekQA pool:
   ALLOW_PAPER_FLYWHEEL_RESEARCH=1 bash solution/scripts/run_paper_flywheel_v1.sh sample-queries ${loop_number} ${current_model}
2. With no Agent/Judge service running, build a fresh index from:
   ${current_model}
   ALLOW_PAPER_FLYWHEEL_RESEARCH=1 bash solution/scripts/run_paper_flywheel_v1.sh index ${loop_number} ${current_model}
3. Serve Agent on both GPUs:
   vllm serve ${agent_model} --port ${agent_port} --tensor-parallel-size 2
4. Collect fresh 10K trajectories while retrieval/FAISS stays on CPU:
   ALLOW_PAPER_FLYWHEEL_RESEARCH=1 bash solution/scripts/run_paper_flywheel_v1.sh collect ${loop_number} ${current_model}
5. Stop the Agent service; serve the judge on both GPUs:
   vllm serve ${judge_model} --port ${judge_port} --tensor-parallel-size 2
6. Build Search->Browse pairs, judge post-browse reasoning, and compute globally normalized paper weights:
   ALLOW_PAPER_FLYWHEEL_RESEARCH=1 bash solution/scripts/run_paper_flywheel_v1.sh build-pairs ${loop_number} ${current_model}
7. Stop the judge; train current retriever for 2 epochs with weighted InfoNCE:
   ALLOW_PAPER_FLYWHEEL_RESEARCH=1 bash solution/scripts/run_paper_flywheel_v1.sh train ${loop_number} ${current_model}
8. Audit lineage, trajectory coverage, sample schema, and exact weight formula:
   bash solution/scripts/run_paper_flywheel_v1.sh audit ${loop_number} ${current_model}
9. For loop $((loop_number+1)), use:
   ${loop_dir}/next_retriever
   and repeat from query sampling; never reuse this loop's sample artifact, index, or trajectories.
EOF
    ;;

  *)
    echo "usage: $0 {preflight|extract-seeds|sample-queries|index|collect|build-pairs|train|audit|plan} [loop_number] [current_model]" >&2
    exit 2
    ;;
esac
