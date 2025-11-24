import torch
from src.models.CLIPs.clip_hba import clip
from pathlib import Path

def load_clip_to_cpu(backbone_name):
    """
    Download (if necessary) and load a CLIP backbone onto CPU memory.

    Parameters
    ----------
    backbone_name : str
        Key referencing the desired CLIP checkpoint in ``clip._MODELS``.

    Returns
    -------
    torch.nn.Module
        The instantiated CLIP model in evaluation mode on the CPU.
    """
    url = clip._MODELS[backbone_name]
    cache_dir = Path.home() / ".cache" / "clip"
    model_path = clip._download(url, str(cache_dir))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model