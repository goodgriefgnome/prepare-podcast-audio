"""Stores large float32 weights as float16 with a Cast back to float32 (download size halves;
the runtime folds the casts and computes in float32).

Tensors whose name contains "keep32" (signal-processing bases) stay float32.

usage: python fp16.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1:3]
m = onnx.load(src)
g = m.graph
keep, casts, n = [], [], 0
for init in g.initializer:
    if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 1024 and "keep32" not in init.name:
        a = numpy_helper.to_array(init)
        h = numpy_helper.from_array(a.astype(np.float16), init.name + "__fp16")
        keep.append(h)
        casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        n += 1
    else:
        keep.append(init)
del g.initializer[:]
g.initializer.extend(keep)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.save(m, dst)
print(f"{src}: {n} tensors to float16")
