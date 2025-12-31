import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext

from src.data.things_dataset import ThingsDataset
from src.data.nod_dataset import NodDataset


def extract_embeddings(model, dataset_name, config, device, logger=None):
    """
    Run a dataset through the model and collect image embeddings.

    Args:
        model: CLIPHBA model instance (already initialized with DoRA layers).
        dataset_name (str): Either 'things' or 'nod'.
        config (dict): Inference configuration containing dataset paths and loader params.
        device (torch.device): Device to perform inference on.
        logger (logging.Logger, optional): Logger for progress messages.

    Returns:
        dict: {
            "image_names": list[str],
            "embeddings": torch.Tensor [N, D],
            "targets" or "categories": labels for each image (dataset-specific)
        }
    """
    if dataset_name not in {"things", "nod"}:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'things' or 'nod'.")

    if "img_dir" not in config or config["img_dir"] is None:
        raise ValueError("config['img_dir'] must be provided for embedding extraction.")

    # Build dataset
    if dataset_name == "things":
        if "img_annotations_file" not in config:
            raise ValueError("config['img_annotations_file'] is required for THINGS embeddings.")

        dataset = ThingsDataset(
            img_annotations_file=config["img_annotations_file"],
            img_dir=config["img_dir"],
        )
    elif dataset_name == "nod":
        if "category_index_file" not in config:
            raise ValueError("config['category_index_file'] is required for NOD embeddings.")

        dataset = NodDataset(
            category_index_file=config["category_index_file"],
            img_dir=config["img_dir"],
            max_images_per_category=config.get("max_images_per_category", 2),
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'things' or 'nod'.")

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 8)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if logger:
        logger.info(f"Extracting embeddings for {len(dataset)} samples from {dataset_name}.")

    all_names = []
    all_embeddings = []
    targets = []
    categories = []

    use_autocast = device.type == "cuda"
    autocast_ctx = torch.amp.autocast if use_autocast else nullcontext

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if dataset_name == "things":
                image_names, images, batch_targets = batch
                targets.append(batch_targets.cpu())
            else:
                image_names, images, batch_categories = batch
                categories.extend(batch_categories)

            all_names.extend(list(image_names))
            images = images.to(device, non_blocking=non_blocking)

            with autocast_ctx(device_type="cuda") if use_autocast else autocast_ctx():
                img_embs = model(
                    images, pos_embedding=model.pos_embedding
                )

            all_embeddings.append(img_embs.cpu())

    outputs = {
        "image_names": all_names,
        "embeddings": torch.cat(all_embeddings, dim=0),
    }

    if dataset_name == "things":
        outputs["targets"] = torch.cat(targets, dim=0)
    else:
        outputs["categories"] = categories

    return outputs