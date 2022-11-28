"""
Extract patch embeddings from CLIP
"""
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from params_proto import PrefixProto, Proto, Flag
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from feature_fields.stego.modules import DinoFeaturizer
from feature_fields.stego.utils import get_transform
from feature_fields.utils import get_source_dir, load_images
from feature_fields.visualize_embeddings import visualize_embeddings


# fmt: off
class DINO_args(PrefixProto):
    # TODO: could consider bigger DINO models
    model_type: str = Proto("vit_small", help="ViT type to use. Valid options: vit_small, vit_base.")
    patch_size: int = Proto(16, help="Patch size for ViT. Valid options: 8, 16.")

    # These are just the DINO model parameters, best to keep to default for now
    dim: int = 70
    use_dropout: bool = True
    feat_type: str = "feat"
    projection_type: str = "nonlinear"
    pretrained_weights: Optional[str] = Proto(dtype=str)
    resolution: int = 512

    # Batch size - batch size of 8 consumes 15.5 out of 24GB on a RTX3090.
    batch_size: int = Proto(1, help="Batch size when evaluating DINO. Keep it small "
                                    "so you don't run out of memory.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Visualize stuff
    visualize_embeddings: bool = Proto(True, help="Whether to visualize the DINO embeddings.")
    visualize_every: int = Proto(5, help="Frequency of images to visualize (e.g. every 10th image).")

    feature_dir: str = Proto("feat", help="Name of directory to store features in.")
    visualize_dir: str = Proto("dino_viz", help="Name of directory to store visualizations in.")
    output_dtype: str = Proto("float16", help="Numpy dtype to use for the DINO output embeddings.")


# @lru_cache(maxsize=None)
def _load_dino() -> Tuple[DinoFeaturizer, Compose]:
    """
    Load DINO and its preprocessing transforms.
    Cache it, so we only have to do it once.
    """
    model = DinoFeaturizer(
        dim=DINO_args.dim,
        patch_size=DINO_args.patch_size,
        feat_type=DINO_args.feat_type,
        model_type=DINO_args.model_type,
        pretrained_weights=DINO_args.pretrained_weights,
        use_dropout=DINO_args.use_dropout,
        projection_type=DINO_args.projection_type,
    )
    model.to(DINO_args.device)
    model.eval()

    # Transform - we need resize, crop, and normalize
    # is_label = False will NOT apply normalization, so we don't want that
    preprocess = get_transform(
        DINO_args.resolution, is_label=False, crop_type="center"
    )

    return model, preprocess


@torch.no_grad()
def get_dino_embeddings(images: List[Image.Image]) -> torch.Tensor:
    """
    Process a list of images and get their patch embeddings with DINO.
    """
    from ml_logger import logger

    # Load DINO
    with logger.time("load_dino_model"):
        model, preprocess = _load_dino()

    # Preprocess each image
    with logger.time("dino_preprocess_images"):
        preprocessed_images = torch.stack([preprocess(image) for image in images])

    dataloader = DataLoader(preprocessed_images, batch_size=DINO_args.batch_size)

    embeddings = []
    # Get DINO embeddings for the images
    with logger.time("get_dino_embeddings"):
        for img_batch in tqdm(dataloader, desc="processing DINO embedding"):
            batch_embeddings, _ = model(img_batch.to(DINO_args.device))
            embeddings.append(batch_embeddings.to('cpu'))

    return torch.cat(embeddings)


def save_dino_embeddings(dataset_path: str) -> str:
    """
    Write DINO embeddings for images in the given dataset and return the path
    to the embeddings pickle.
    """
    from ml_logger import logger

    logger.log_params(DINO_args=vars(DINO_args))
    logger.log_text("""
    charts:
    - type: image
      glob: "**/*.png"
    """, ".charts.yml", True, True)

    source_dir = get_source_dir(dataset_path)
    images, image_paths = load_images(source_dir, minimum_images=2)
    logger.print(f"Loaded {len(images)} images from {source_dir}")

    # Create the feature directory if it doesn't exist
    feature_dir = os.path.join(dataset_path, DINO_args.feature_dir)
    os.makedirs(feature_dir, exist_ok=True)

    # Load the images from disk and get the CLIP embeddings
    embeddings = get_dino_embeddings(images).numpy()
    embeddings = embeddings.astype(DINO_args.output_dtype)
    logger.print(f"embedding shape = {embeddings.shape}")

    # Write embeddings to disk
    with logger.time("write_dino_embeddings"):
        embeddings_fname = os.path.join(feature_dir, "dino.npy")
        np.save(embeddings_fname, embeddings)
    logger.print(f"Wrote DINO embeddings to {embeddings_fname}")

    # Visualize embeddings for debugging purposes
    if DINO_args.visualize_embeddings:
        with logger.time("visualize_dino_embeddings"):
            # Reshape to num_image x num_patches x feature_dim
            visualize_embeddings(
                image_paths,
                images,
                embeddings,
                preprocess=_load_dino()[1],
                visualize_every=DINO_args.visualize_every,
                alpha=0.9
            )

    return embeddings_fname


if __name__ == "__main__":
    from ml_logger import instr

    dataset = "$DATASETS/custom_nerf/Autotraj"
    thunk = instr(save_dino_embeddings)
    thunk(dataset_path=os.path.expandvars(dataset))
