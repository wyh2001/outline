#!/usr/bin/env bash

set -euo pipefail

ort_lib_dir="${GITHUB_WORKSPACE}/${ORT_DIR}/lib"

export LD_LIBRARY_PATH="${ort_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OUTLINE_TEST_ORT_DYLIB="${ort_lib_dir}/libonnxruntime.so"
export ORT_DYLIB_PATH="${ort_lib_dir}/libonnxruntime.so"

cargo test --verbose --no-default-features --features ort-load-dynamic --lib
cargo test --verbose --test runtime_dynamic --test runtime_dynamic_env --no-default-features --features ort-load-dynamic -- --ignored
