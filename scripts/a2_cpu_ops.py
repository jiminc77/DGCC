"""CPU-only real-checkpoint I/O callables for the A-2 harness (lower-bound measurement).

Loads the actual G6b lock-ckpt of v1_s0 to CPU; patch op = one A0 interchange +
one A1 projected splice on real-sized tensors (256-dim h_p batch of 200), which
is the per-pair CPU cost shape of the Q-ranking series.
"""
import sys
sys.path.insert(0, "/home/simx2204/Workspaces/DGCC/src")
import torch

CKPT = "/home/simx2204/Workspaces/DGCC/outputs/models/sprint_t2_v1_s0/ckpt_0250880.pt"

def load():
    return torch.load(CKPT, map_location="cpu", weights_only=False)

def unload(payload):
    del payload

_h_r = torch.randn(200, 256)
_h_d = torch.randn(200, 256)

def patch():
    from dgcc.analysis.sprint_patching import a0_interchange
    return a0_interchange(_h_r, _h_d)
