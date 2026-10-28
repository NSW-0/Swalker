"""GLTNet wrapper: silhouette sequence -> 4096-d gait signature."""

from pathlib import Path

import numpy as np
import torch

from app.ml.combined_glfe_mstfe import GLFEMSTFEModel

IMG_H, IMG_W = 64, 44
FEATURE_DIM = 4096


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class GaitEmbedder:
    """Loads the checkpoint once. Always in eval mode; inference always under no_grad."""

    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.device = device
        self.checkpoint_name = Path(checkpoint_path).name

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state = ckpt["model"]
        # Older checkpoints don't store n_classes; the classifier's row count is the same number.
        n_classes = ckpt.get("n_classes") or state["classifier.weight"].shape[0]

        model = GLFEMSTFEModel(n_parts=8, branch_c=128, head_out_c=256, local_stride=3, n_classes=n_classes)
        self.model = model.to(device)
        # strict: an architecture that doesn't match the checkpoint must fail here,
        # not produce plausible-looking wrong vectors.
        self.model.load_state_dict(state, strict=True)
        # BatchNorm everywhere: a gallery encoded in train mode once made the model look broken.
        self.model.eval()

    def embed(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: (T, 64, 44) float in [0, 1]. Returns float32 (4096,)."""
        if sequence.ndim != 3 or sequence.shape[1:] != (IMG_H, IMG_W):
            raise ValueError(f"expected (T, {IMG_H}, {IMG_W}), got {sequence.shape}")
        x = torch.from_numpy(np.ascontiguousarray(sequence, dtype=np.float32))
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, T, 64, 44)
        with torch.no_grad():
            feat, _logits = self.model(x)  # logits are the training classifier: never used
        return feat[0].float().cpu().numpy()
