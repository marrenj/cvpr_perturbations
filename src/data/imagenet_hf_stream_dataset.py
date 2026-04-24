"""
Streaming ImageNet-1k dataset backed by HuggingFace `datasets`.

Uses HF's streaming mode (no upfront ~160 GB download) and yields samples
as an ``IterableDataset``. Mirrors the (image_name, image_tensor,
class_label) tuple contract used by ``ImagenetDataset``.

Shuffling
---------
PyTorch's ``DataLoader(shuffle=True)`` is rejected for ``IterableDataset``
so shuffling happens inside the dataset via HF's two-level scheme:
    * Shards (parquet files) are permuted each epoch.
    * A rolling buffer of ``shuffle_buffer_size`` samples produces per-batch
      randomness within the shard stream.

Per-epoch randomness requires reseeding — the trainer MUST call
``dataset.set_epoch(epoch_idx)`` before each epoch. Combined with
``persistent_workers=False`` on the DataLoader, the updated ``_epoch``
value propagates to freshly forked workers at every epoch boundary.

Auth
----
ImageNet-1k on HF is gated. ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``)
must be set and the license accepted on the Hub.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from src.data.imagenet_dataset import IMAGENET_MEAN, IMAGENET_STD


IMAGENET_NUM_CLASSES = 1000


def _build_transform(split: str, img_size: int) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImagenetHFStreamDataset(IterableDataset):
    """
    Streaming ImageNet-1k from HuggingFace parquet shards.

    Args:
        split: ``'train'`` or ``'val'`` (mapped to HF's ``validation``).
        num_samples: Nominal dataset size. Returned by ``__len__`` so
            trainer code using ``len(loader.dataset)`` keeps working.
        hf_dataset_name: HF repo id, default ``'ILSVRC/imagenet-1k'``.
        cache_dir: HF_DATASETS_CACHE location (node-local recommended).
        img_size: Spatial resolution (default 224).
        shuffle_buffer_size: Buffer size for HF's streaming shuffle.
        seed: Base shuffle seed; effective seed is ``base_seed + epoch``.
        image_col / label_col: Column names in the HF parquet schema.
    """

    def __init__(
        self,
        split: str = "train",
        num_samples: Optional[int] = None,
        hf_dataset_name: str = "ILSVRC/imagenet-1k",
        cache_dir: Optional[str] = None,
        img_size: int = 224,
        shuffle_buffer_size: int = 10_000,
        seed: int = 0,
        image_col: str = "image",
        label_col: str = "label",
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if num_samples is None or num_samples <= 0:
            raise ValueError(
                "num_samples must be provided and > 0 for a streaming dataset"
            )

        self.split = split
        self._hf_split = "train" if split == "train" else "validation"
        self.num_samples = int(num_samples)
        self.hf_dataset_name = hf_dataset_name
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.base_seed = seed
        self.image_col = image_col
        self.label_col = label_col

        self._epoch = 0
        self.transform = _build_transform(split, img_size)

        print(
            f"ImageNet {split} (HF stream): nominal {self.num_samples:,} samples, "
            f"{IMAGENET_NUM_CLASSES} classes, source={hf_dataset_name}"
        )

    def __len__(self) -> int:
        return self.num_samples

    @property
    def num_classes(self) -> int:
        return IMAGENET_NUM_CLASSES

    def set_epoch(self, epoch: int) -> None:
        """
        Update the epoch counter used to reseed HF's streaming shuffle.

        Must be called before each epoch. Relies on
        ``persistent_workers=False`` so that workers are re-forked each
        epoch and pick up the updated value.
        """
        self._epoch = int(epoch)

    def _open_hf_stream(self):
        from datasets import load_dataset

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        return load_dataset(
            self.hf_dataset_name,
            split=self._hf_split,
            streaming=True,
            cache_dir=self.cache_dir,
            token=token,
        )

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor, int]]:
        dataset = self._open_hf_stream()

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.split == "train":
            dataset = dataset.shuffle(
                seed=self.base_seed + self._epoch,
                buffer_size=self.shuffle_buffer_size,
            )

        local_idx = 0
        for sample in dataset:
            try:
                image = sample[self.image_col]
                label = int(sample[self.label_col])

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image_tensor = self.transform(image)
            except Exception as exc:
                print(
                    f"[ImagenetHFStreamDataset] worker={worker_id} "
                    f"skipped corrupt sample (local_idx={local_idx}): {exc}"
                )
                local_idx += 1
                continue

            image_name = f"{self.split}_e{self._epoch}_w{worker_id}_{local_idx}.jpg"
            local_idx += 1
            yield image_name, image_tensor, label
