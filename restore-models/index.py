"""Writes index.json: for each model file, its parts (in order), total size and SHA-256.

usage: python index.py <dir with *.onnx.partNN files>
"""
import hashlib, json, os, re, sys

d = sys.argv[1]
models = {}
for name in sorted(os.listdir(d)):
    m = re.fullmatch(r"(.+\.onnx)\.part(\d+)", name)
    if m:
        models.setdefault(m.group(1), []).append(name)
index = {}
for model, parts in models.items():
    h = hashlib.sha256()
    size = 0
    for p in parts:
        with open(os.path.join(d, p), "rb") as f:
            data = f.read()
        h.update(data)
        size += len(data)
    index[model] = {"parts": parts, "size": size, "sha256": h.hexdigest()}
with open(os.path.join(d, "index.json"), "w") as f:
    json.dump(index, f, indent=1)
