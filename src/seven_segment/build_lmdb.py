import argparse
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import lmdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_KEY = b"__metadata__"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# Conservative multiplier for metadata + pickle overhead
DEFAULT_HEADROOM_FACTOR = 1.3

# Fallback cap (only used if estimation fails)
DEFAULT_MAX_MAP_SIZE_GB = 50


def estimate_map_size_bytes(dataset_dirs: list[str], headroom: float) -> int:
    """
    Estimate LMDB map size based on actual file sizes on disk.
    This is OS-agnostic and avoids relying on sparse file behaviour.
    """
    total_bytes = 0

    for img_path, _ in iter_samples(dataset_dirs):
        try:
            total_bytes += os.path.getsize(img_path)
        except OSError:
            continue

    # Apply headroom for:
    # - pickle overhead
    # - LMDB page structure
    # - any future additions
    estimated = int(total_bytes * headroom)

    return estimated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iter_samples(dataset_dirs: list[str]):
    """
    Yield (image_path, class_id) for every image found across all dataset dirs.
    Class id is inferred from the subfolder name (must be an integer 0-15).
    """
    for d in dataset_dirs:
        for class_name in sorted(os.listdir(d)):
            class_dir = os.path.join(d, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                class_id = int(class_name)
            except ValueError:
                print(f"  Skipping non-integer subfolder: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    yield os.path.join(class_dir, fname), class_id


def count_samples(dataset_dirs: list[str]) -> int:
    return sum(1 for _ in iter_samples(dataset_dirs))


# ---------------------------------------------------------------------------
# LMDB schema
#
# Each sample is stored as a pickle of:
#   {"image": np.ndarray (HxWx3 uint8, BGR), "label": int}
#
# Raw BGR is stored so the preprocessor retains full flexibility over how to
# convert or threshold — colour channel selection, HSV binarisation, etc.
# Baking grayscale or binarisation into storage would prevent iterating on
# those decisions without rebuilding the LMDB from source.
#
# Key: zero-padded index bytes, e.g. b"00000001"
#
# Metadata (key = METADATA_KEY) is a pickle of a dict:
#   {
#     "num_samples": int,
#     "class_counts": {class_id: count},
#     "source_dirs": [str],
#   }
# ---------------------------------------------------------------------------


def build_lmdb(
    dataset_dirs: list[str],
    output_path: str,
    map_size_gb: int,
) -> None:
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output path '{output_path}' already exists. "
            "Delete it or choose a different path."
        )

    print("Counting samples...")
    total = count_samples(dataset_dirs)
    print(f"Found {total} images across {len(dataset_dirs)} director(ies).")

    if map_size_gb is not None:
        map_size = map_size_gb * (1024**3)
        print(f"Using user-specified map size: {map_size_gb} GB")
    else:
        print("Estimating map size from dataset...")
        map_size = estimate_map_size_bytes(dataset_dirs, DEFAULT_HEADROOM_FACTOR)

        # Optional safety cap
        max_bytes = DEFAULT_MAX_MAP_SIZE_GB * (1024**3)
        if map_size > max_bytes:
            print(
                f"Estimated size exceeds {DEFAULT_MAX_MAP_SIZE_GB} GB cap. "
                "Clamping to avoid OS-specific allocation issues."
            )
            map_size = max_bytes

        print(f"Estimated map size: {map_size / (1024**3):.2f} GB")
    class_counts = defaultdict(int)

    env = lmdb.open(output_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, (img_path, class_id) in enumerate(
            tqdm(iter_samples(dataset_dirs), total=total, desc="Writing LMDB")
        ):
            # Store raw BGR — full colour preserved so the preprocessor can
            # exploit hue/saturation to separate digit colour from background noise
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  Warning: could not read {img_path}, skipping.")
                continue

            key = f"{idx:08d}".encode()
            value = pickle.dumps({"image": img, "label": class_id}, protocol=4)
            txn.put(key, value)
            class_counts[class_id] += 1

        # Write metadata so the Dataset class can find num_samples without scanning keys
        metadata = {
            "num_samples": total,
            "class_counts": dict(class_counts),
            "source_dirs": [os.path.abspath(d) for d in dataset_dirs],
        }
        txn.put(METADATA_KEY, pickle.dumps(metadata, protocol=4))

    env.close()
    print(f"\nLMDB written to: {output_path}")
    print(f"Total samples: {total}")
    for c in sorted(class_counts):
        print(f"  class {c:2d}: {class_counts[c]}")


# ---------------------------------------------------------------------------
# PyTorch Dataset (included here for reference — copy into your training code)
# ---------------------------------------------------------------------------


class LMDBScoreDataset:
    """
    PyTorch-compatible dataset backed by the LMDB file produced by this script.

    Usage:
        dataset = LMDBScoreDataset("path/to/output.lmdb", transform=your_transform)
        loader  = DataLoader(dataset, batch_size=32, num_workers=4)

    transform should include SevenSegmentScorePreprocessor (or a torchvision
    Compose wrapping it) so preprocessing is applied on-the-fly per sample.
    This means preprocessing changes never require rebuilding the LMDB.

    The LMDB environment is opened lazily per worker to avoid issues with
    forked processes sharing a file handle — a common source of DataLoader bugs.
    """

    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self._env = None

        # Read metadata without keeping the environment open
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            meta = pickle.loads(txn.get(METADATA_KEY))
        env.close()

        self.num_samples = meta["num_samples"]
        self.class_counts = meta["class_counts"]

    def _get_env(self):
        # Open lazily so each DataLoader worker gets its own handle after fork
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False
            )
        return self._env

    def __len__(self):
        return self.num_samples

    def read_all_labels(self) -> list[int]:
        """Read all labels without leaving the environment open."""
        import pickle

        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        labels = []
        with env.begin() as txn:
            for idx in range(self.num_samples):
                key = f"{idx:08d}".encode()
                sample = pickle.loads(txn.get(key))
                labels.append(sample["label"])
        env.close()
        return labels

    def __getitem__(self, idx):
        env = self._get_env()
        key = f"{idx:08d}".encode()
        with env.begin() as txn:
            sample = pickle.loads(txn.get(key))

        image = sample["image"]  # np.ndarray HxWx3 uint8 BGR
        label = sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack one or more ImageFolder dataset directories into a single LMDB file"
    )
    parser.add_argument(
        "output_path",
        help="Path for the output LMDB directory (must not already exist)",
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        help="One or more ImageFolder-layout dataset directories to pack",
    )
    parser.add_argument(
        "--map-size-gb",
        type=int,
        default=None,
        help=(
            "LMDB map size in GB. If not provided, it will be estimated from dataset size. "
            "Explicitly setting this is recommended on systems with strict disk allocation (e.g. Windows)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    build_lmdb(args.dataset_dirs, args.output_path, args.map_size_gb)
