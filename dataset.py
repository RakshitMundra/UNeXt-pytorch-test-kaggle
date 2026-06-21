import os

import cv2
import numpy as np
import torch
import torch.utils.data

# Let the DataLoader workers handle parallelism; otherwise OpenCV spawns its own
# threads in every worker process and oversubscribes the CPU (CPU pegged, GPU idle).
cv2.setNumThreads(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes,
                 transform=None, cache=False):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
            cache (bool): If True, decode every image/mask once into a single
                contiguous uint8 array held in this (main) process. DataLoader
                workers then share it via copy-on-write fork instead of each
                re-decoding files every epoch. Only use on a pre-resized dataset
                where all images share one size and the set fits in RAM as uint8.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        self.cache = cache
        self.img_cache = None
        self.mask_cache = None
        if cache:
            self._build_cache()

    def _read_img(self, img_id):
        return cv2.imread(os.path.join(self.img_dir, img_id + '_sat' + self.img_ext))

    def _read_mask(self, img_id):
        return cv2.imread(os.path.join(self.mask_dir, img_id + '_mask' + self.mask_ext),
                          cv2.IMREAD_GRAYSCALE)

    def _build_cache(self):
        # Probe the first sample to size the contiguous arrays. All images must
        # share one (H, W) -- true for a pre-resized dataset.
        first_img = self._read_img(self.img_ids[0])
        if first_img is None:
            raise FileNotFoundError(
                'cache=True but could not read %s_sat%s' % (self.img_ids[0], self.img_ext))
        h, w = first_img.shape[:2]
        n = len(self.img_ids)

        # Single allocations -> one buffer each, fully shareable by forked workers.
        self.img_cache = np.empty((n, h, w, 3), dtype=np.uint8)
        self.mask_cache = np.empty((n, h, w, 1), dtype=np.uint8)
        for i, img_id in enumerate(self.img_ids):
            img = self._read_img(img_id) if i else first_img
            mask = self._read_mask(img_id)
            if img.shape[:2] != (h, w) or mask.shape[:2] != (h, w):
                raise ValueError(
                    'cache=True requires uniform size; %s differs. Pre-resize first.' % img_id)
            self.img_cache[i] = img
            self.mask_cache[i] = mask[..., None]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        if self.cache:
            # Index the shared RAM cache; copy so albumentations / the float cast
            # below never write back into the copy-on-write buffer.
            img = self.img_cache[idx].copy()
            mask = self.mask_cache[idx].copy()
        else:
            # Read at native resolution. For speed/RAM, pre-resize the dataset once
            # with preprocess_resize.py and point DATA_ROOT at the cached copy, so
            # these files are already small (the Resize in the transform is then a
            # no-op). Do NOT use IMREAD_REDUCED_* here: it would halve already-resized
            # files and force an upscale.
            img = cv2.imread(os.path.join(self.img_dir, img_id + '_sat' + self.img_ext))

            mask = cv2.imread(os.path.join(self.mask_dir, img_id + '_mask' + self.mask_ext),
                              cv2.IMREAD_GRAYSCALE)[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Keep as uint8 CHW. The float cast + normalization are done on the GPU
        # in the training loop: smaller host->device transfer (1 byte vs 4) and
        # the per-pixel float work is offloaded from the CPU workers.
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        mask = np.ascontiguousarray(mask.transpose(2, 0, 1))

        return img, mask, {'img_id': img_id}
