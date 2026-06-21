"""One-time dataset pre-resize.

Reads paired <stem>_sat.* / <stem>_mask.* files from a source directory,
resizes both to a fixed square size, and writes them as PNG into a destination
directory (keeping the _sat / _mask naming).

Why: training re-decodes + re-resizes the same large tiles every epoch, which
pins the CPU and inflates RAM (OOM). Doing it once up front means each worker
later decodes a tiny already-sized file -> lower CPU and lower RAM.

Example (Kaggle):
    python preprocess_resize.py \
        --src /kaggle/input/<slug>/train \
        --dst /kaggle/working/data512 \
        --size 512

Then train against the cache:
    DATA_ROOT=/kaggle/working python train.py --dataset data512 \
        --img_ext .png --mask_ext .png ...
"""
import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm

# This is a single-threaded batch job; let OpenCV use the cores freely here.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True,
                        help='source dir containing *_sat.* and *_mask.* files')
    parser.add_argument('--dst', required=True,
                        help='destination dir for resized *_sat.png / *_mask.png')
    parser.add_argument('--size', default=512, type=int,
                        help='output square size in pixels (default: 512)')
    parser.add_argument('--sat_suffix', default='_sat', help='image filename suffix')
    parser.add_argument('--mask_suffix', default='_mask', help='mask filename suffix')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.dst, exist_ok=True)

    sats = sorted(glob(os.path.join(args.src, '*' + args.sat_suffix + '.*')))
    if not sats:
        raise SystemExit(f'No *{args.sat_suffix}.* files found in {args.src}')
    print(f'Found {len(sats)} images -> resizing to {args.size}x{args.size} in {args.dst}')

    size = (args.size, args.size)
    written, skipped = 0, 0
    for sat_path in tqdm(sats):
        base = os.path.basename(sat_path)
        stem = base.rsplit(args.sat_suffix, 1)[0]

        mask_matches = glob(os.path.join(args.src, stem + args.mask_suffix + '.*'))
        if not mask_matches:
            skipped += 1
            continue

        img = cv2.imread(sat_path)
        mask = cv2.imread(mask_matches[0], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skipped += 1
            continue

        # INTER_AREA is best for downscaling photos; NEAREST keeps the mask's
        # discrete label values intact (no interpolated in-between values).
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(args.dst, stem + args.sat_suffix + '.png'), img)
        cv2.imwrite(os.path.join(args.dst, stem + args.mask_suffix + '.png'), mask)
        written += 1

    print(f'Done: wrote {written} pairs, skipped {skipped} (missing/unreadable mask).')


if __name__ == '__main__':
    main()
