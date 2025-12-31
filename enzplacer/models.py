from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

import torch
import torch.nn as nn


class LayerNormNet(nn.Module):
    """
    Projector used in EnzPlacer training: ESM-1b mean embedding (1280-d) -> projected embedding (out_dim).

    Architecture:
      Linear(1280 -> hidden) + LayerNorm + Dropout + ReLU
      Linear(hidden -> hidden) + LayerNorm + Dropout + ReLU
      Linear(hidden -> out_dim)
    """
    def __init__(self, hidden_dim: int, out_dim: int, drop_out: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(1280, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


@dataclass(frozen=True)
class ProjectorSpec:
    hidden_dim: int
    out_dim: int


def infer_projector_spec(state_dict: Dict[str, torch.Tensor]) -> ProjectorSpec:
    """
    Infer hidden/out dims from a LayerNormNet-style checkpoint.
    Expected keys: fc1.weight, fc2.weight, fc3.weight.
    """
    if "fc1.weight" not in state_dict or "fc3.weight" not in state_dict:
        raise ValueError("Checkpoint does not look like a LayerNormNet state_dict (missing fc1.weight/fc3.weight).")

    hidden_dim = int(state_dict["fc1.weight"].shape[0])
    out_dim = int(state_dict["fc3.weight"].shape[0])
    return ProjectorSpec(hidden_dim=hidden_dim, out_dim=out_dim)


def load_projector(ckpt_path: str, device: Union[str, torch.device] = "cpu") -> Tuple[nn.Module, ProjectorSpec]:
    """
    Load a projector checkpoint saved via torch.save(model.state_dict(), path).

    Returns (model, spec). The model is set to eval() and moved to device.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError("Unsupported checkpoint format. Expected a state_dict (dict[str, Tensor]).")

    # Handle DataParallel-style prefixes if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    spec = infer_projector_spec(sd)
    model = LayerNormNet(hidden_dim=spec.hidden_dim, out_dim=spec.out_dim)
    model.load_state_dict(sd, strict=True)
    model.to(device=device)
    model.eval()
    return model, spec
