#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_recurrent_scale_campaign_gpu.sh \
    --source-commit <full-sha> \
    --repository-url <git-remote> \
    --workspace-parent <absolute-dir> \
    --venv-parent <absolute-dir> \
    --output-parent <absolute-dir> [options]

Creates a new content-addressed checkout, venv, and output tree without
modifying an existing project checkout. It runs CUDA/source/dependency/tests
preflight, pins runtime provenance, proves mid-training exact-auxiliary crash
resume parity with populated Adam state, then starts all 8 x 3 resumable cells
and aggregate
analysis in tmux.

Options:
  --bootstrap-python <command>     Python used only to create a new venv
  --device <torch-device>          Default: cuda:0
  --rollout-workers <count>        Per arm; default: 8
  --counterfactual-workers <count> Per arm; default: 8
  --evaluation-workers <count>     Per arm; default: 8
  --uv-command <absolute-path>     Optional uv binary outside PATH
  --session-name <name>            Default: evosim-scale-<sha12>
  --foreground                     Run the campaign worker in this terminal

Use an SSH-configured Git remote or another credential-free argument. Do not
put access tokens in --repository-url.
EOF
}

scale_mode="launch"
scale_source_commit=""
scale_repository_url=""
scale_workspace_parent=""
scale_venv_parent=""
scale_output_parent=""
scale_checkout=""
scale_venv=""
scale_campaign_output=""
scale_bootstrap_python="python3"
scale_uv_command=""
scale_device="cuda:0"
scale_rollout_workers=8
scale_counterfactual_workers=8
scale_evaluation_workers=8
scale_session_name=""
scale_foreground=false

while (($#)); do
  case "$1" in
    --worker)
      scale_mode="worker"
      shift
      ;;
    --source-commit)
      scale_source_commit="${2:?missing --source-commit value}"
      shift 2
      ;;
    --repository-url)
      scale_repository_url="${2:?missing --repository-url value}"
      shift 2
      ;;
    --workspace-parent)
      scale_workspace_parent="${2:?missing --workspace-parent value}"
      shift 2
      ;;
    --venv-parent)
      scale_venv_parent="${2:?missing --venv-parent value}"
      shift 2
      ;;
    --output-parent)
      scale_output_parent="${2:?missing --output-parent value}"
      shift 2
      ;;
    --checkout)
      scale_checkout="${2:?missing --checkout value}"
      shift 2
      ;;
    --venv)
      scale_venv="${2:?missing --venv value}"
      shift 2
      ;;
    --campaign-output)
      scale_campaign_output="${2:?missing --campaign-output value}"
      shift 2
      ;;
    --bootstrap-python)
      scale_bootstrap_python="${2:?missing --bootstrap-python value}"
      shift 2
      ;;
    --uv-command)
      scale_uv_command="${2:?missing --uv-command value}"
      shift 2
      ;;
    --device)
      scale_device="${2:?missing --device value}"
      shift 2
      ;;
    --rollout-workers)
      scale_rollout_workers="${2:?missing --rollout-workers value}"
      shift 2
      ;;
    --counterfactual-workers)
      scale_counterfactual_workers="${2:?missing --counterfactual-workers value}"
      shift 2
      ;;
    --evaluation-workers)
      scale_evaluation_workers="${2:?missing --evaluation-workers value}"
      shift 2
      ;;
    --session-name)
      scale_session_name="${2:?missing --session-name value}"
      shift 2
      ;;
    --foreground)
      scale_foreground=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

fail() {
  echo "scale launcher error: $*" >&2
  exit 1
}

require_positive_integer() {
  local label="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$label must be a positive integer"
}

require_safe_absolute_parent() {
  local label="$1"
  local value="$2"
  [[ "$value" == /* ]] || fail "$label must be an absolute path"
  [[ "$value" != "/" ]] || fail "$label cannot be the filesystem root"
}

verify_clean_exact_checkout() {
  local checkout="$1"
  local expected_commit="$2"
  [[ -d "$checkout/.git" ]] || fail "isolated checkout is not a Git worktree"
  local observed_commit
  observed_commit="$(git -C "$checkout" rev-parse HEAD)"
  [[ "$observed_commit" == "$expected_commit" ]] || \
    fail "isolated checkout HEAD differs from the requested commit"
  [[ -z "$(git -C "$checkout" status --porcelain --untracked-files=all)" ]] || \
    fail "isolated checkout is dirty; refusing to mutate or train"
}

verify_venv() {
  local venv="$1"
  [[ -f "$venv/pyvenv.cfg" && -x "$venv/bin/python" ]] || \
    fail "versioned venv is incomplete"
  "$venv/bin/python" -m pip check
}

run_cuda_smoke() {
  local python="$1"
  local device="$2"
  "$python" -c '
import sys
import torch

device = torch.device(sys.argv[1])
if device.type != "cuda" or not torch.cuda.is_available():
    raise SystemExit("CUDA device is required for the GPU campaign")
x = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16)
y = x @ x.T
if not torch.isfinite(y).all().item():
    raise SystemExit("CUDA smoke produced non-finite output")
torch.cuda.synchronize(device)
print(
    "cuda_smoke_ok",
    f"torch={torch.__version__}",
    f"cuda_build={torch.version.cuda}",
    f"device={torch.cuda.get_device_name(device)}",
)
' "$device"
}

prepare_preregistration_and_provenance() {
  local checkout="$1"
  local python="$2"
  local campaign_output="$3"
  local source_commit="$4"
  local preregistration="$campaign_output/preregistration.json"
  local runtime_provenance="$campaign_output/runtime-provenance.json"

  mkdir -p "$campaign_output"
  if [[ ! -f "$preregistration" ]]; then
    "$python" -m \
      evolution_sim.cli.mind_v3_public_recurrent_ippo_scale_campaign \
      preregister \
      --source-commit "$source_commit" \
      --output "$preregistration"
  fi
  "$python" -c '
import json
import sys

from evolution_sim.mind.recurrent_scale_campaign import (
    validate_recurrent_scale_campaign_preregistration,
)

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
validate_recurrent_scale_campaign_preregistration(payload)
if payload["source"]["commit"] != sys.argv[2]:
    raise SystemExit("existing preregistration source commit drifted")
' "$preregistration" "$source_commit"

  "$python" -m \
    evolution_sim.cli.mind_v3_public_recurrent_ippo_runtime_provenance \
    --preregistration "$preregistration" \
    --output "$runtime_provenance" \
    --device "$scale_device" \
    --rollout-workers "$scale_rollout_workers" \
    --counterfactual-workers "$scale_counterfactual_workers" \
    --evaluation-workers "$scale_evaluation_workers"
}

run_cuda_training_resume_smoke() {
  local python="$1"
  local preregistration="$2"
  local runtime_provenance="$3"
  local output_root="$4"
  local preregistration_digest
  preregistration_digest="$($python -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["exact_digest"])
' "$preregistration")"
  "$python" -m \
    evolution_sim.cli.mind_v3_public_recurrent_ippo_cuda_training_smoke \
    --preregistration "$preregistration" \
    --expected-preregistration-digest "$preregistration_digest" \
    --runtime-provenance "$runtime_provenance" \
    --output-root "$output_root" \
    --device "$scale_device" \
    --rollout-workers "$scale_rollout_workers" \
    --counterfactual-workers "$scale_counterfactual_workers" \
    --evaluation-workers "$scale_evaluation_workers"
}

run_campaign_worker() {
  verify_clean_exact_checkout "$scale_checkout" "$scale_source_commit"
  verify_venv "$scale_venv"
  local python="$scale_venv/bin/python"
  local preregistration="$scale_campaign_output/preregistration.json"
  local runtime_provenance="$scale_campaign_output/runtime-provenance.json"
  local reports_root="$scale_campaign_output/runs"
  local analysis="$scale_campaign_output/analysis.json"
  local logs="$scale_campaign_output/logs"

  export PYTHONPATH="$scale_checkout/python"
  export PYTHONHASHSEED=0
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  mkdir -p "$reports_root" "$logs"
  run_cuda_smoke "$python" "$scale_device"
  prepare_preregistration_and_provenance \
    "$scale_checkout" \
    "$python" \
    "$scale_campaign_output" \
    "$scale_source_commit"
  [[ -f "$runtime_provenance" ]] || fail "runtime provenance was not pinned"
  run_cuda_training_resume_smoke \
    "$python" \
    "$preregistration" \
    "$runtime_provenance" \
    "$scale_campaign_output/preflight/cuda-training-smoke"

  local preregistration_digest
  preregistration_digest="$($python -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["exact_digest"])
' "$preregistration")"

  local -a learner_seeds
  local -a arms
  mapfile -t learner_seeds < <($python -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
for value in payload["seed_contract"]["learner_seeds"]:
    print(value)
' "$preregistration")
  mapfile -t arms < <($python -c '
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
for value in payload["arms"]["order"]:
    print(value)
' "$preregistration")
  [[ "${#learner_seeds[@]}" -eq 8 ]] || fail "preregistration lacks 8 learners"
  [[ "${#arms[@]}" -eq 3 ]] || fail "preregistration lacks 3 arms"

  local learner_seed
  local arm
  for learner_seed in "${learner_seeds[@]}"; do
    local -a arm_processes=()
    for arm in "${arms[@]}"; do
      local run_id="scale-v1-learner-${learner_seed}-${arm}"
      echo "starting_or_resuming $run_id"
      (
        "$python" -m \
          evolution_sim.cli.mind_v3_public_recurrent_ippo_scale_campaign \
          run-arm \
          --preregistration "$preregistration" \
          --expected-preregistration-digest "$preregistration_digest" \
          --learner-seed "$learner_seed" \
          --arm "$arm" \
          --output-root "$reports_root" \
          --runtime-provenance "$runtime_provenance" \
          --device "$scale_device" \
          --rollout-workers "$scale_rollout_workers" \
          --counterfactual-workers "$scale_counterfactual_workers" \
          --evaluation-workers "$scale_evaluation_workers" \
          --resume 2>&1 | tee -a "$logs/${run_id}.log"
      ) &
      arm_processes+=("$!")
    done
    local learner_failed=false
    local arm_process
    for arm_process in "${arm_processes[@]}"; do
      if ! wait "$arm_process"; then
        learner_failed=true
      fi
    done
    [[ "$learner_failed" == false ]] || \
      fail "one or more paired arms failed for learner $learner_seed"
  done

  "$python" -m \
    evolution_sim.cli.mind_v3_public_recurrent_ippo_scale_campaign \
    aggregate \
    --preregistration "$preregistration" \
    --reports-root "$reports_root" \
    --runtime-provenance "$runtime_provenance" \
    --output "$analysis" 2>&1 | tee -a "$logs/aggregate.log"
  echo "scale_campaign_complete analysis=$analysis"
}

[[ "$scale_source_commit" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]] || \
  fail "--source-commit must be a full lowercase Git hash"
require_positive_integer "--rollout-workers" "$scale_rollout_workers"
require_positive_integer \
  "--counterfactual-workers" "$scale_counterfactual_workers"
require_positive_integer "--evaluation-workers" "$scale_evaluation_workers"
[[ "$scale_device" == cuda:* ]] || fail "GPU launcher requires a cuda:<index> device"

if [[ "$scale_mode" == "worker" ]]; then
  require_safe_absolute_parent "--checkout" "$scale_checkout"
  require_safe_absolute_parent "--venv" "$scale_venv"
  require_safe_absolute_parent "--campaign-output" "$scale_campaign_output"
  run_campaign_worker
  exit 0
fi

[[ -n "$scale_repository_url" ]] || fail "--repository-url is required"
require_safe_absolute_parent "--workspace-parent" "$scale_workspace_parent"
require_safe_absolute_parent "--venv-parent" "$scale_venv_parent"
require_safe_absolute_parent "--output-parent" "$scale_output_parent"
command -v git >/dev/null || fail "git is required"
command -v tmux >/dev/null || fail "tmux is required"
command -v "$scale_bootstrap_python" >/dev/null || \
  fail "bootstrap Python command is unavailable"

scale_short_commit="${scale_source_commit:0:12}"
scale_checkout="${scale_workspace_parent%/}/evolution-sim-scale-${scale_short_commit}"
scale_venv="${scale_venv_parent%/}/${scale_short_commit}"
scale_campaign_output="${scale_output_parent%/}/${scale_short_commit}"
if [[ -z "$scale_session_name" ]]; then
  scale_session_name="evosim-scale-${scale_short_commit}"
fi
[[ "$scale_session_name" =~ ^[A-Za-z0-9_.-]+$ ]] || \
  fail "--session-name contains unsupported characters"

mkdir -p "$scale_workspace_parent" "$scale_venv_parent" "$scale_output_parent"
if [[ ! -e "$scale_checkout" ]]; then
  git clone --filter=blob:none --no-checkout \
    "$scale_repository_url" "$scale_checkout"
  git -C "$scale_checkout" fetch --no-tags origin "$scale_source_commit"
  git -C "$scale_checkout" checkout --detach "$scale_source_commit"
fi
verify_clean_exact_checkout "$scale_checkout" "$scale_source_commit"

if [[ ! -e "$scale_venv" ]]; then
  "$scale_bootstrap_python" -m venv "$scale_venv"
  if [[ -n "$scale_uv_command" ]]; then
    [[ -x "$scale_uv_command" ]] || fail "--uv-command is not executable"
    "$scale_uv_command" pip install \
      --python "$scale_venv/bin/python" \
      --requirement "$scale_checkout/requirements-mind-ml.txt"
  elif command -v uv >/dev/null; then
    uv pip install \
      --python "$scale_venv/bin/python" \
      --requirement "$scale_checkout/requirements-mind-ml.txt"
  elif "$scale_bootstrap_python" -m uv --version >/dev/null 2>&1; then
    "$scale_bootstrap_python" -m uv pip install \
      --python "$scale_venv/bin/python" \
      --requirement "$scale_checkout/requirements-mind-ml.txt"
  else
    "$scale_venv/bin/python" -m pip install \
      --requirement "$scale_checkout/requirements-mind-ml.txt"
  fi
fi
verify_venv "$scale_venv"

scale_logical_cpus="$("$scale_venv/bin/python" -c 'import os; print(os.cpu_count() or 0)')"
scale_max_workers="$scale_rollout_workers"
if ((scale_counterfactual_workers > scale_max_workers)); then
  scale_max_workers="$scale_counterfactual_workers"
fi
if ((scale_evaluation_workers > scale_max_workers)); then
  scale_max_workers="$scale_evaluation_workers"
fi
scale_concurrent_worker_budget=$((3 * scale_max_workers))
if ((scale_logical_cpus > 0 && scale_concurrent_worker_budget > scale_logical_cpus)); then
  fail "three concurrent arms request more workers than logical CPUs"
fi

export PYTHONPATH="$scale_checkout/python"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
run_cuda_smoke "$scale_venv/bin/python" "$scale_device"
(
  cd "$scale_checkout"
  "$scale_venv/bin/python" -m unittest \
    python.tests.test_recurrent_runtime_provenance \
    python.tests.test_recurrent_scale_campaign \
    python.tests.test_recurrent_scale_execution
)
prepare_preregistration_and_provenance \
  "$scale_checkout" \
  "$scale_venv/bin/python" \
  "$scale_campaign_output" \
  "$scale_source_commit"
run_cuda_training_resume_smoke \
  "$scale_venv/bin/python" \
  "$scale_campaign_output/preregistration.json" \
  "$scale_campaign_output/runtime-provenance.json" \
  "$scale_campaign_output/preflight/cuda-training-smoke"

scale_worker=(
  "$scale_checkout/scripts/run_recurrent_scale_campaign_gpu.sh"
  --worker
  --source-commit "$scale_source_commit"
  --checkout "$scale_checkout"
  --venv "$scale_venv"
  --campaign-output "$scale_campaign_output"
  --device "$scale_device"
  --rollout-workers "$scale_rollout_workers"
  --counterfactual-workers "$scale_counterfactual_workers"
  --evaluation-workers "$scale_evaluation_workers"
)
if [[ "$scale_foreground" == true ]]; then
  "${scale_worker[@]}"
  exit 0
fi

if tmux has-session -t "$scale_session_name" 2>/dev/null; then
  echo "scale campaign session already exists: $scale_session_name"
  echo "attach with: tmux attach -t $scale_session_name"
  exit 0
fi
mkdir -p "$scale_campaign_output/logs"
printf -v scale_worker_command '%q ' "${scale_worker[@]}"
printf -v scale_campaign_log '%q' "$scale_campaign_output/logs/campaign.log"
tmux new-session -d -s "$scale_session_name" \
  "${scale_worker_command} >>${scale_campaign_log} 2>&1"
echo "scale campaign launched: $scale_session_name"
echo "attach with: tmux attach -t $scale_session_name"
echo "output root: $scale_campaign_output"
