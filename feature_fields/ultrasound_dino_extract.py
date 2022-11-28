"""
Extract patch embeddings from CLIP
"""
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import glob

from feature_fields.visualize_embeddings import visualize_embeddings
from feature_fields.dino_extract import get_dino_embeddings
import torchvision.transforms as T


if __name__ == "__main__":
    image_paths = sorted(glob.glob('/home/yban/Works/Data/data_simple_liver/data/images/us/*.png'))
    PIL_Images = [Image.open(x).convert('RGB') for x in image_paths]

    transform_preprocess = T.Compose([T.Resize(size=512, interpolation=T.InterpolationMode.NEAREST, max_size=None, antialias=None),
                            T.CenterCrop(size=(512, 512)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                            )
    
    emb = get_dino_embeddings(PIL_Images)

    MAX_ITER = 10
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 67
    Bi_RGB_STD = 3
    BGR_MEAN = np.array([104.008, 116.669, 122.675])

    
    visualize_embeddings(
                image_paths,
                PIL_Images,
                emb,
                preprocess=transform_preprocess,
                visualize_every=1,
                alpha=0.8)
    print(emb.shape)
    print('Done.')
