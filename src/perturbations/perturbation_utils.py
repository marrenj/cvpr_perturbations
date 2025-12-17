import torch


class PerturbationStrategy:
    def __init__(self, perturb_epoch: int, perturb_length: int, perturb_seed: int):
        # Calculate the perturbation window 
        self.start_epoch = perturb_epoch
        self.end_epoch = perturb_epoch + perturb_length - 1
        self.perturb_seed = perturb_seed

    def is_active_epoch(self, epoch_idx: int) -> bool:
        """Check if the perturbation should be applied in the given epoch (0-indexed)."""
        return self.start_epoch <= epoch_idx <= self.end_epoch

    def apply_to_batch(self, images, targets, device, epoch_idx: int, batch_idx: int):
        """Apply the perturbation to the batch (if active); return (images, targets)."""
        raise NotImplementedError


class TargetNoisePerturbation(PerturbationStrategy):
    def __init__(self, perturb_epoch, perturb_length, perturb_seed, target_mean=None, target_std=None):
        super().__init__(perturb_epoch, perturb_length, perturb_seed)
        self.target_mean = target_mean
        self.target_std = target_std

    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets  # no change outside perturbation window

        if self.target_mean is not None and self.target_std is not None:
            mean = self.target_mean.to(device=targets.device, dtype=targets.dtype)
            std = self.target_std.to(device=targets.device, dtype=targets.dtype)
        else:
            raise ValueError("target_mean and target_std must be provided for TargetNoisePerturbation")

        gen = torch.Generator(device=device)
        gen.manual_seed(self.perturb_seed + (epoch_idx + 1) * 1000 + batch_idx)

        noise = torch.randn(targets.shape, device=device, dtype=targets.dtype, generator=gen)
        random_targets = noise * std + mean
        return images, random_targets


class LabelShufflePerturbation(PerturbationStrategy):
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets
        gen = torch.Generator(device=device)
        gen.manual_seed(self.perturb_seed + (self.start_epoch+1) * 1000 + batch_idx)
        shuffled = shuffle_targets(targets, generator=gen)
        return images, shuffled


class ImageNoisePerturbation(PerturbationStrategy):
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets
        # Set global seed for this batch (to ensure reproducibility across devices if needed)
        torch.manual_seed(self.perturb_seed + (self.start_epoch+1) * 1000 + batch_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(self.perturb_seed + (self.start_epoch+1) * 1000 + batch_idx)
        # Create noise of the same shape as images (this will be drawn from N~(0,1), same as the normalized images)
        noise = torch.randn_like(images)
        return noise, targets


class UniformImagePerturbation(PerturbationStrategy):
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        if not self.is_active_epoch(epoch_idx):
            return images, targets
        uniform = torch.ones_like(images) * 0.5
        return uniform, targets


class NoPerturbation(PerturbationStrategy):
    def apply_to_batch(self, images, targets, device, epoch_idx, batch_idx):
        return images, targets


def shuffle_targets(targets, generator=None):
    """
    Shuffle the target vectors along the batch dimension using the provided generator.
    """
    batch_size = targets.shape[0]
    perm = torch.randperm(batch_size, device=targets.device, generator=generator)
    return targets[perm]


# Choose the appropriate perturbation strategy
def choose_perturbation_strategy(
    perturb_type,
    perturb_epoch,
    perturb_length,
    perturb_seed,
    target_mean=None,
    target_std=None,
):
    if perturb_type == 'random_target':
        perturb_strategy = TargetNoisePerturbation(
            perturb_epoch=perturb_epoch,
            perturb_length=perturb_length,
            perturb_seed=perturb_seed,
            target_mean=target_mean,
            target_std=target_std,
        )
    elif perturb_type == 'label_shuffle':
        perturb_strategy = LabelShufflePerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    elif perturb_type == 'image_noise':
        perturb_strategy = ImageNoisePerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    elif perturb_type == 'uniform_images':
        perturb_strategy = UniformImagePerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    elif perturb_type == 'None':
        perturb_strategy = NoPerturbation(perturb_epoch=perturb_epoch, perturb_length=perturb_length, perturb_seed=perturb_seed)
    else:
        raise ValueError(f"Perturbation type {perturb_type} not supported")
    return perturb_strategy