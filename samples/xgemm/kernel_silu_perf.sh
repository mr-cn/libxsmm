#!/usr/bin/env bash
###############################################################################
# Compares fused SiLU against several baselines:
# 1) unfused GEMM plus a standalone unary SiLU
# 2) fused Sigmoid
# 3) fused ReLU
###############################################################################
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
REPS=${REPS:-4000}
BENCHMARK_DURATION=${BENCHMARK_DURATION:-0.2}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

resolve_bin() {
  local stem=$1
  if [ -x "${stem}.exe" ]; then
    printf '%s.exe' "${stem}"
  else
    printf '%s' "${stem}"
  fi
}

GEMM_BIN=${GEMM_BIN:-$(resolve_bin "${HERE}/gemm_kernel")}
FUSED_BIN=${FUSED_BIN:-$(resolve_bin "${HERE}/gemm_kernel_fused")}
UNARY_BIN=${UNARY_BIN:-$(resolve_bin "${HERE}/../eltwise/eltwise_unary_simple")}

extract_gemm_runtime() {
  printf '%s\n' "$1" | sed -n 's/^\([0-9.e+-][0-9.e+-]*\)s for [Ll][Ii][Bb][Xx][Ss][Mm][Mm]$/\1/p' | tail -n 1
}

extract_unary_ips() {
  printf '%s\n' "$1" | sed -n 's/^Iterations\/s  : \([0-9.e+-][0-9.e+-]*\)$/\1/p' | tail -n 1
}

assert_parsed() {
  local value=$1
  local name=$2
  local output=$3
  if [ -z "${value}" ]; then
    printf 'Failed to parse %s\n' "${name}" >&2
    printf '%s\n' "${output}" >&2
    exit 1
  fi
}

format_row() {
  awk \
    -v label="$1" \
    -v gemm_only="$2" \
    -v relu="$3" \
    -v sigmoid="$4" \
    -v silu="$5" \
    -v unary="$6" \
    'BEGIN {
      unfused = gemm_only + unary;
      printf "%-18s %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %10.4f %10.4f %10.4f %10.4f\n",
        label,
        gemm_only * 1e6,
        relu * 1e6,
        sigmoid * 1e6,
        silu * 1e6,
        unary * 1e6,
        unfused * 1e6,
        silu / gemm_only,
        silu / relu,
        silu / sigmoid,
        unfused / silu;
    }'
}

run_case() {
  local label=$1
  local a_dt=$2
  local b_dt=$3
  local comp_dt=$4
  local c_dt=$5
  local m=$6
  local n=$7
  local k=$8
  local br_type=$9
  local br_count=${10}
  local beta=${11}
  local binary_postop=${12}

  local gemm_output
  local relu_output
  local sigmoid_output
  local silu_output
  local unary_output
  local gemm_runtime
  local relu_runtime
  local sigmoid_runtime
  local silu_runtime
  local unary_ips
  local gemm_iter
  local relu_iter
  local sigmoid_iter
  local silu_iter
  local unary_iter

  if [ "${binary_postop}" -eq 0 ]; then
    gemm_output=$(OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${GEMM_BIN}" \
      "${a_dt}" "${b_dt}" "${comp_dt}" "${c_dt}" \
      "${m}" "${n}" "${k}" "${m}" "${k}" "${m}" \
      1 "${beta}" 0 0 0 0 0 0 0 nopf "${br_type}" "${br_count}" 0 "${REPS}")
  else
    gemm_output=$(OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${FUSED_BIN}" \
      "${a_dt}" "${b_dt}" "${comp_dt}" "${c_dt}" \
      "${m}" "${n}" "${k}" "${m}" "${k}" "${m}" \
      1 "${beta}" 0 0 0 0 0 0 0 nopf "${br_type}" "${br_count}" 0 "${REPS}" 0 "${binary_postop}")
  fi

  relu_output=$(OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${FUSED_BIN}" \
    "${a_dt}" "${b_dt}" "${comp_dt}" "${c_dt}" \
    "${m}" "${n}" "${k}" "${m}" "${k}" "${m}" \
    1 "${beta}" 0 0 0 0 0 0 0 nopf "${br_type}" "${br_count}" 0 "${REPS}" 0 "${binary_postop}" 1)

  sigmoid_output=$(OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${FUSED_BIN}" \
    "${a_dt}" "${b_dt}" "${comp_dt}" "${c_dt}" \
    "${m}" "${n}" "${k}" "${m}" "${k}" "${m}" \
    1 "${beta}" 0 0 0 0 0 0 0 nopf "${br_type}" "${br_count}" 0 "${REPS}" 0 "${binary_postop}" 3)

  silu_output=$(OMP_NUM_THREADS="${OMP_NUM_THREADS}" "${FUSED_BIN}" \
    "${a_dt}" "${b_dt}" "${comp_dt}" "${c_dt}" \
    "${m}" "${n}" "${k}" "${m}" "${k}" "${m}" \
    1 "${beta}" 0 0 0 0 0 0 0 nopf "${br_type}" "${br_count}" 0 "${REPS}" 0 "${binary_postop}" 4)

  unary_output=$(BENCHMARK_DURATION="${BENCHMARK_DURATION}" "${UNARY_BIN}" \
    18 0 "${c_dt}" "${comp_dt}" "${c_dt}" "${m}" "${n}" "${m}" "${m}" 0)

  gemm_runtime=$(extract_gemm_runtime "${gemm_output}")
  relu_runtime=$(extract_gemm_runtime "${relu_output}")
  sigmoid_runtime=$(extract_gemm_runtime "${sigmoid_output}")
  silu_runtime=$(extract_gemm_runtime "${silu_output}")
  unary_ips=$(extract_unary_ips "${unary_output}")

  assert_parsed "${gemm_runtime}" "GEMM runtime" "${gemm_output}"
  assert_parsed "${relu_runtime}" "ReLU runtime" "${relu_output}"
  assert_parsed "${sigmoid_runtime}" "Sigmoid runtime" "${sigmoid_output}"
  assert_parsed "${silu_runtime}" "SiLU runtime" "${silu_output}"
  assert_parsed "${unary_ips}" "unary iterations/s" "${unary_output}"

  gemm_iter=$(awk -v rt="${gemm_runtime}" -v reps="${REPS}" 'BEGIN { printf "%.12e", rt / reps }')
  relu_iter=$(awk -v rt="${relu_runtime}" -v reps="${REPS}" 'BEGIN { printf "%.12e", rt / reps }')
  sigmoid_iter=$(awk -v rt="${sigmoid_runtime}" -v reps="${REPS}" 'BEGIN { printf "%.12e", rt / reps }')
  silu_iter=$(awk -v rt="${silu_runtime}" -v reps="${REPS}" 'BEGIN { printf "%.12e", rt / reps }')
  unary_iter=$(awk -v ips="${unary_ips}" 'BEGIN { printf "%.12e", 1.0 / ips }')

  format_row "${label}" "${gemm_iter}" "${relu_iter}" "${sigmoid_iter}" "${silu_iter}" "${unary_iter}"
}

printf 'Using GEMM_BIN=%s\n' "${GEMM_BIN}"
printf 'Using FUSED_BIN=%s\n' "${FUSED_BIN}"
printf 'Using UNARY_BIN=%s\n' "${UNARY_BIN}"
printf 'REPS=%s BENCHMARK_DURATION=%s OMP_NUM_THREADS=%s\n\n' "${REPS}" "${BENCHMARK_DURATION}" "${OMP_NUM_THREADS}"
printf "%-18s %12s %12s %12s %12s %12s %12s %10s %10s %10s %10s\n" \
  "case" "gemm[us]" "relu[us]" "sigm[us]" "silu[us]" "unary[us]" "unfused[us]" "s/gemm" "s/relu" "s/sigm" "u/silu"

run_case "f32_edge"         F32  F32  F32 F32  23  17  19 nobr   1 1 0
run_case "f32_square64"     F32  F32  F32 F32  64  64  64 nobr   1 1 0
run_case "f32_square128"    F32  F32  F32 F32 128 128 128 nobr   1 1 0
run_case "f32_addrbr64x5"   F32  F32  F32 F32  64  64  64 addrbr 5 1 0
run_case "f32_colbias64"    F32  F32  F32 F32  64  64  64 nobr   1 1 1

if [ "${RUN_BF16:-0}" -ne 0 ]; then
  run_case "bf16_square64"    BF16 BF16 F32 BF16 64 64 64 nobr   1 1 0
  run_case "bf16_addrbr64x5"  BF16 BF16 F32 BF16 64 64 64 addrbr 5 1 0
  run_case "bf16_colbias64"   BF16 BF16 F32 BF16 64 64 64 nobr   1 1 1
fi
