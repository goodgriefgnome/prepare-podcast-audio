"""Loads Resemble Enhance's enhancer (stage 2) for export, without its training dependencies."""
import sys, types, torch
import os
# RE_SRC: a checkout of resemble-enhance; RE_RUN: its enhancer_stage2 weights folder.
sys.path.insert(0, os.environ["RE_SRC"])
# deepspeed is only needed for training
import importlib.abc, importlib.machinery

class _Stub(types.ModuleType):
    """Any attribute is a dummy class; deepspeed is only used for training."""
    __path__ = []
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        return _Stub(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _Finder())
from pathlib import Path
from resemble_enhance.enhancer.hparams import HParams
from resemble_enhance.enhancer.enhancer import Enhancer
from resemble_enhance.inference import remove_weight_norm_recursively

import scipy.optimize
from resemble_enhance.enhancer.lcfm import cfm as _cfm

def _mapping(t, n=4):
    # Same as upstream; fsolve returns an array, which newer numpy won't turn into a float.
    h = lambda t, a: (a**t - 1) / (a - 1)
    a = float(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0)[0])
    return h(t, a=a)

_cfm.Solver.exponential_decay_mapping = staticmethod(_mapping)

RUN = Path(os.environ["RE_RUN"])

def load():
    hp = HParams.load(RUN)
    m = Enhancer(hp)
    sd = torch.load(RUN / "ds/G/default/mp_rank_00_model_states.pt", map_location="cpu")["module"]
    m.load_state_dict(sd)
    m.eval()
    remove_weight_norm_recursively(m)
    return m

if __name__ == "__main__":
    m = load()
    print(m.hp)
    for name, mod in m.named_children():
        print(name, sum(p.numel() for p in mod.parameters()) / 1e6, "M")
