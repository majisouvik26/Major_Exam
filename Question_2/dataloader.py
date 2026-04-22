import os
import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _read_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def collect_image_mask_paths(data_root="data"):
    image_dir = os.path.join(data_root, "CameraRGB")
    mask_dir = os.path.join(data_root, "CameraMask")

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"Expected folders not found: {image_dir} and/or {mask_dir}"
        )

    image_files = {
        file_name
        for file_name in os.listdir(image_dir)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
    }
    mask_files = {
        file_name
        for file_name in os.listdir(mask_dir)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    common_files = sorted(image_files.intersection(mask_files))
    if not common_files:
        raise ValueError("No matching image-mask filenames found.")

    image_paths = [os.path.join(image_dir, file_name) for file_name in common_files]
    mask_paths = [os.path.join(mask_dir, file_name) for file_name in common_files]
    return image_paths, mask_paths


def train_test_split_paths(image_paths, mask_paths, train_ratio=0.8, seed=42):
    if len(image_paths) != len(mask_paths):
        raise ValueError("Image and mask path counts do not match.")

    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_idx = int(num_samples * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    return train_images, train_masks, test_images, test_masks


def build_color_to_class_map(mask_paths, num_classes=23):
    colors = set()
    total_masks = len(mask_paths)
    for idx, mask_path in enumerate(mask_paths, start=1):
        mask_rgb = _read_rgb_image(mask_path)
        unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
        colors.update(map(tuple, unique_colors.tolist()))
        if idx % 100 == 0 or idx == total_masks:
            print(
                f"Color-map preprocessing: {idx}/{total_masks} masks scanned | unique_colors={len(colors)}",
                flush=True,
            )

    if len(colors) > num_classes:
        raise ValueError(
            f"Found {len(colors)} unique colors, which is more than configured classes ({num_classes})."
        )

    if len(colors) < num_classes:
        print(
            f"Warning: dataset has {len(colors)} observed classes, fewer than configured {num_classes}.",
            flush=True,
        )

    sorted_colors = sorted(colors)
    return {color: class_idx for class_idx, color in enumerate(sorted_colors)}


def _load_or_build_color_to_class_map(mask_paths, data_root, num_classes=23):
    cache_path = os.path.join(data_root, "color_to_class_map.json")

    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        color_to_class_map = {
            tuple(map(int, key.split(","))): int(val) for key, val in raw_map.items()
        }
        if len(color_to_class_map) > num_classes:
            raise ValueError(
                f"Cached class map has {len(color_to_class_map)} classes, expected at most {num_classes}."
            )
        if len(color_to_class_map) < num_classes:
            print(
                f"Warning: cached class map has {len(color_to_class_map)} observed classes; model still uses {num_classes} output classes.",
                flush=True,
            )
        print(f"Loaded cached color map from: {cache_path}", flush=True)
        return color_to_class_map

    print("Building color-to-class map from masks...", flush=True)
    color_to_class_map = build_color_to_class_map(mask_paths, num_classes=num_classes)
    serializable_map = {
        f"{color[0]},{color[1]},{color[2]}": class_idx
        for color, class_idx in color_to_class_map.items()
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable_map, f, indent=2)
    print(f"Saved color map cache to: {cache_path}", flush=True)
    return color_to_class_map


def _encode_mask(mask_rgb, color_to_class_map):
    h, w, _ = mask_rgb.shape
    encoded_mask = np.zeros((h, w), dtype=np.int64)

    flat_keys = (
        (mask_rgb[..., 0].astype(np.int64) << 16)
        | (mask_rgb[..., 1].astype(np.int64) << 8)
        | mask_rgb[..., 2].astype(np.int64)
    )

    key_to_class = {
        (color[0] << 16) | (color[1] << 8) | color[2]: class_idx
        for color, class_idx in color_to_class_map.items()
    }

    for key, class_idx in key_to_class.items():
        encoded_mask[flat_keys == key] = class_idx

    return encoded_mask


class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, color_to_class_map, image_size=(128, 96)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.color_to_class_map = color_to_class_map
        self.image_size = image_size

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Image and mask path counts do not match.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = _read_rgb_image(self.image_paths[idx])
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        mask_rgb = _read_rgb_image(self.mask_paths[idx])
        mask_rgb = cv2.resize(mask_rgb, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = _encode_mask(mask_rgb, self.color_to_class_map)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        return img, mask


def create_dataloaders(
    data_root="data",
    batch_size=8,
    image_size=(128, 96),
    train_ratio=0.8,
    seed=42,
    num_workers=0,
):
    image_paths, mask_paths = collect_image_mask_paths(data_root=data_root)
    color_to_class_map = _load_or_build_color_to_class_map(
        mask_paths,
        data_root=data_root,
        num_classes=23,
    )

    train_images, train_masks, test_images, test_masks = train_test_split_paths(
        image_paths=image_paths,
        mask_paths=mask_paths,
        train_ratio=train_ratio,
        seed=seed,
    )

    train_dataset = CityscapesDataset(
        train_images,
        train_masks,
        color_to_class_map=color_to_class_map,
        image_size=image_size,
    )
    test_dataset = CityscapesDataset(
        test_images,
        test_masks,
        color_to_class_map=color_to_class_map,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, color_to_class_map