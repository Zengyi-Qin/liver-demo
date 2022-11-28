import glob
import os
from functools import lru_cache
from typing import List, Tuple

from PIL import Image

_VALID_IMAGE_EXTENSIONS = "jpg jpeg png JPG JPEG PNG".split(' ')


@lru_cache
def get_images_in_dir(image_dir: str, recursive: bool = True) -> List[str]:
    """Finds and returns all images in the given directory."""
    images = []
    for ext in _VALID_IMAGE_EXTENSIONS:
        if recursive:
            glob_pattern = f"{image_dir}/**/*.{ext}"
        else:
            glob_pattern = f"{image_dir}/*.{ext}"
        images.extend(glob.glob(glob_pattern, recursive=recursive))
    return images


def get_source_dir(dataset_path: str) -> str:
    """
    Returns the path to the source directory of the given dataset.

    We assume the images have been written to the source/ or images/
    directory (the latter in case of CLIPort datasets) in a flat
    structure as expected by Fast NeRF.
    """
    dataset_path = os.path.expandvars(dataset_path)
    source_dir = os.path.join(dataset_path, "source")
    if os.path.isdir(source_dir):
        return source_dir

    # Check if dataset follows CLIPort directory structure
    source_dir = os.path.join(dataset_path, "images")
    transforms_json = os.path.join(dataset_path, "transforms.json")
    if os.path.exists(transforms_json) and os.path.isdir(source_dir):
        print(f"Detected CLIPort dataset for {dataset_path}")
        return source_dir

    raise ValueError(f"Could not determine source image directory for {dataset_path}")


def load_images(
        image_dir: str, minimum_images: int = 1, recursive: bool = False
) -> Tuple[List[Image.Image], List[str]]:
    """
    Loads all images in the given directory.
    If recursive is True, then all subdirectories are searched for images.

    Returns a tuple of with the list of images and the list of image paths.
    """
    image_paths = get_images_in_dir(image_dir, recursive=recursive)
    if len(image_paths) < minimum_images:
        raise ValueError(f"Found {len(image_paths)} images in {image_dir}. Need at least {minimum_images} images.")

    images = [Image.open(image_path) for image_path in image_paths]
    return images, image_paths
