#!/usr/bin/env bash

set -euo pipefail

ort_lib_dir="${GITHUB_WORKSPACE}/${ORT_DIR}/lib"

export LD_LIBRARY_PATH="${ort_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PKG_CONFIG_PATH="${GITHUB_WORKSPACE}/${ORT_PKGCONFIG_DIR}"

mkdir -p "${PKG_CONFIG_PATH}"
cat >"${PKG_CONFIG_PATH}/libonnxruntime.pc" <<EOF
prefix=${GITHUB_WORKSPACE}/${ORT_DIR}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: onnxruntime
Description: ONNX runtime
URL: https://github.com/microsoft/onnxruntime
Version: ${ORT_VERSION}
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
EOF

cargo test --verbose --no-default-features --features ort-pkg-config --lib
