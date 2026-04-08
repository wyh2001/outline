#!/usr/bin/env bash

set -euo pipefail

ort_lib_dir="${GITHUB_WORKSPACE}/${ORT_DIR}/lib"

export LD_LIBRARY_PATH="${ort_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ORT_LIB_LOCATION="${ort_lib_dir}"
export ORT_PREFER_DYNAMIC_LINK="1"

cargo test --verbose --no-default-features --lib
