from pathlib import Path
from typing import List

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Normalize


def visualize_embeddings(
    image_paths: List[str],
    images: List[Image.Image],
    embeddings: np.ndarray,
    preprocess: Compose,
    visualize_every: int,
    alpha: float = 0.5,
    should_plot: bool = False,
):
    """
    Save visualized embeddings. We show the pre-processed image
    without normalization and the embeddings projected using PCA into
    three components (RGB).

    We blend the resulting images and save it to disk and show the
    combined plot if show_plot is True.
    """
    from ml_logger import logger
    from sklearn.decomposition import PCA

    # Embeddings takes in the torch format.
    embeddings = rearrange(embeddings, "n c h w -> n (h w) c")

    if visualize_every < 1:
        raise ValueError("visualize_every must be >= 1")

    # Flatten to shape num_patches * clip_embedding_dimensionality
    og_viz_embeddings = embeddings[::visualize_every]
    num_patches = og_viz_embeddings.shape[0] * og_viz_embeddings.shape[1]
    viz_embeddings = og_viz_embeddings.reshape(num_patches, -1)

    # PCA with RGB components
    pca = PCA(n_components=3)
    pca.fit(viz_embeddings)
    embeddings_rgb = pca.transform(viz_embeddings)

    # Normalize by min-max of each R G B channel (it doesn't matter too much)
    rgb_max = embeddings_rgb.max(0)
    rgb_min = embeddings_rgb.min(0)
    normalized_embeddings_rgb = (embeddings_rgb - rgb_min) / (rgb_max - rgb_min)
    normalized_embeddings_rgb = normalized_embeddings_rgb.reshape(
        *og_viz_embeddings.shape[:2], 3
    )

    # Blend the pre-processed image with the PCA embedding and save to disk
    blended_images = []
    for idx, (image_path, image, embedding_rgb) in enumerate(
        zip(
            image_paths[::visualize_every],
            images[::visualize_every],
            normalized_embeddings_rgb,
        )
    ):
        # Apply all but last normalize pre-process transform
        assert isinstance(preprocess.transforms[-1], Normalize)
        for t in preprocess.transforms[:-1]:
            image = t(image)
        image = image.permute(1, 2, 0).cpu().numpy()

        # Reshape embeddings to pre-processed image shape
        num_patches_ax = np.sqrt(embedding_rgb.shape[0]).astype(int)
        embedding_rgb = embedding_rgb.reshape(num_patches_ax, num_patches_ax, 3)
        scaling_factor = (image.shape[0] / num_patches_ax).astype(int)
        embedding_rgb = np.kron(
            embedding_rgb, np.ones((scaling_factor, scaling_factor, 1))
        )
        # Plot embedding as a transparent layer on image
        image_255 = (image * 255).astype(np.uint8)
        embedding_rgb_255 = (embedding_rgb * 255).astype(np.uint8)
        blended = Image.blend(
            Image.fromarray(image_255), Image.fromarray(embedding_rgb_255), alpha=alpha
        )
        blended_images.append(blended)
        if should_plot:
            plt.figure()
            plt.imshow(blended)
            plt.show()
        else:
            # Save to ml-logger
            path = logger.save_image(
                blended, f"figures/{Path(image_path).stem}_viz.png"
            )
            logger.print(f"Saved image at {path}.", color="yellow")

    return blended_images
