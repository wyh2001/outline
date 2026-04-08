import json
import subprocess
import sys

metadata = json.loads(
    subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--locked"], text=True)
)
root = next(pkg for pkg in metadata["packages"] if pkg["name"] == "outline-core")
ort = next(pkg for pkg in metadata["packages"] if pkg["name"] == "ort")
feature_map = root["features"]
ort_defaults = set(ort["features"]["default"])

enabled = {
    feature
    for dep in root["dependencies"]
    if dep["name"] == "ort"
    for feature in dep["features"]
}

pending = list(feature_map["default"])
seen = set()

while pending:
    item = pending.pop()
    if item.startswith("ort/"):
        enabled.add(item.split("/", 1)[1])
        continue
    if item.startswith("dep:") or item in seen:
        continue
    if item in feature_map:
        seen.add(item)
        pending.extend(feature_map[item])

missing = sorted(ort_defaults - enabled)
if missing:
    print(f"missing mirrored ort default features: {', '.join(missing)}", file=sys.stderr)
    print(f"ort defaults: {', '.join(sorted(ort_defaults))}", file=sys.stderr)
    print(f"enabled here: {', '.join(sorted(enabled))}", file=sys.stderr)
    raise SystemExit(1)

print(f"ort default features mirrored: {', '.join(sorted(ort_defaults))}")
