#!/usr/bin/env bash

set -euo pipefail

mkdir -p "${ORT_ROOT}"

archive_path="${ORT_ROOT}/${ORT_ARCHIVE}"
extract_dir="${ORT_ROOT}/onnxruntime-linux-x64-${ORT_VERSION}"
library_path="${ORT_DIR}/lib/libonnxruntime.so"

if [ ! -f "${archive_path}" ]; then
	curl -L --fail --retry 3 -o "${archive_path}" "${ORT_URL}"
fi

echo "${ORT_SHA256}  ${archive_path}" | sha256sum -c -

if [ ! -e "${library_path}" ]; then
	rm -rf "${ORT_DIR}" "${extract_dir}"
	tar -xzf "${archive_path}" -C "${ORT_ROOT}"
	mv -T "${extract_dir}" "${ORT_DIR}"
fi

tar -czf "${ORT_ASSET_ARCHIVE}" -C "${ORT_ROOT}" onnxruntime
