"""Import ImageReward 1.5 with its three relocated Transformers helpers.

Transformers moved these helpers to pytorch_utils; they keep the same API.
No scoring weights, architecture, preprocessing or forward formula is changed.
"""


def import_image_reward():
    from importlib.metadata import version

    from transformers import modeling_utils, pytorch_utils

    if version("image-reward") != "1.5" or version("transformers") != "4.57.1":
        raise RuntimeError(
            "ImageReward compatibility adapter requires the pinned evaluation dependencies"
        )
    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
    import ImageReward

    return ImageReward
