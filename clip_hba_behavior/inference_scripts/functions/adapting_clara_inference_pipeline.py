#!/usr/bin/env python3
"""
Multi-GPU Things checkpoint evaluator
———————————————
* Spawns one long-lived worker process per GPU (safe for CUDA)
* Uses `spawn` start-method so no CUDA context is inherited
* Evaluates Things dataset using pairwise cosine similarity and Pearson correlation
* Saves individual results per checkpoint and a final summary
"""

import os
import re
import numpy as np
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import pandas as pd
import torch.multiprocessing as mp
import argparse
import logging
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Dataset, DataLoader

from src.model import CLIP
from config import VIT_B_32_CONFIG

from run_all_datacomp_eval_debug import (
    get_image_transform,
    CustomTokenizer,
    RetrievalDataset,
    load_clip_model_from_checkpoint,
    image_captions_collate_fn,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThingsDataset(Dataset):
    def __init__(self, image_dir: str, ordered_image_filenames: List[str], transform=None):
        self.image_dir = Path(image_dir)
        self.ordered_image_filenames = ordered_image_filenames
        self.transform = transform
        
    def __len__(self):
        return len(self.ordered_image_filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        img_filename = self.ordered_image_filenames[idx]
        img_path = self.image_dir / img_filename
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logger.error(f"Image file not found: {img_path}. Please check image_dir and filenames.")
            # For robustness, one might return a placeholder or skip, but for now, error out.
            raise
        
        if self.transform:
            image = self.transform(image)
            
        # Derive prompt text from the filename's stem
        concept_stem = Path(img_filename).stem 
        prompt_text = concept_stem.replace('_', ' ') # Example: "file_name_example" -> "file name example"
            
        return image, img_filename, prompt_text

def discover_checkpoints(ckpt_dir: str, start: int, end: int) -> List[Tuple[int, str]]:
    """Return [(epoch, ckpt_path), …] within [start, end]."""
    ckpt_files = sorted(
        glob(f"{ckpt_dir}/monitor_val_cifar100_epoch_acc1-epoch_epoch=*.ckpt")
    )
    out: List[Tuple[int, str]] = []
    for ckpt in ckpt_files:
        m = re.search(r"epoch_epoch=(\d+)\.ckpt", ckpt)
        if m:
            epoch = int(m.group(1))
            if start <= epoch <= end:
                out.append((epoch, ckpt))
    return out

def load_clip_model_from_checkpoint(checkpoint_path: str, device: int) -> CLIP:
    """Load CLIP model from checkpoint."""
    logger.info(f"[GPU {device}] Loading model from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            logger.error(f"[GPU {device}] Invalid state dict in checkpoint")
            return None

        # Create model
        model = CLIP(config=VIT_B_32_CONFIG, random_text_encoder=True)
        
        # Clean state dict
        cleaned_state_dict = OrderedDict()
        prefix = 'model.'
        has_prefix = any(k.startswith(prefix) for k in state_dict.keys())
        
        if has_prefix:
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    cleaned_state_dict[k[len(prefix):]] = v
                else:
                    cleaned_state_dict[k] = v
        else:
            cleaned_state_dict = state_dict

        # Load weights
        model.load_state_dict(cleaned_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        logger.error(f"[GPU {device}] Error loading checkpoint: {e}")
        return None

def compute_rdm_and_correlation(embeddings, reference_rdm):
    # Compute RDM as 1 - pairwise cosine similarity
    normalized_embeddings = F.normalize(embeddings, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    model_rdm = similarity_matrix.cpu().numpy()
    
    # Flatten the upper triangular part (excluding diagonal)
    n = model_rdm.shape[0]
    model_rdm_flat = model_rdm[np.triu_indices(n, k=1)]
    reference_rdm_flat = reference_rdm[np.triu_indices(n, k=1)]
    
    # Compute Pearson correlation
    correlation, p_value = pearsonr(model_rdm_flat, reference_rdm_flat)
    spearman_correlation, spearman_p_value = spearmanr(model_rdm_flat, reference_rdm_flat)
    return correlation, p_value, spearman_correlation, spearman_p_value

def evaluate_things(
    model: CLIP, 
    dataloader: DataLoader, 
    reference_rdm: np.ndarray, 
    device: int
) -> Dict:
    """
    Extract image and text embeddings simultaneously, compute their RDMs, 
    and correlate with a reference RDM.
    """
    model.eval()
    tokenizer = CustomTokenizer(model)

    all_image_embeddings_list = []
    all_text_embeddings_list = []
    all_img_filenames_from_loader = [] 
    
    logger.info(f"[GPU {device}] Starting simultaneous image and text embedding extraction...")
    with torch.no_grad():
        for images_batch, img_filenames_batch, prompt_texts_batch in tqdm(
            dataloader, 
            desc=f"[GPU {device}] Extracting image/text embeddings", 
            unit="batch"
        ):
            images_batch = images_batch.to(device)
            
            # Image embeddings
            image_embeds = model.encode_image(images_batch) # Normalization in compute_rdm_and_correlation
            all_image_embeddings_list.append(image_embeds.cpu())
            
            # Text embeddings
            tokenized_prompts = tokenizer(list(prompt_texts_batch)).to(device)
            text_embeds = model.encode_text(tokenized_prompts, normalize=True) 
            all_text_embeddings_list.append(text_embeds.cpu())
            
            all_img_filenames_from_loader.extend(img_filenames_batch)
            
    all_image_embeddings_tensor = torch.cat(all_image_embeddings_list, dim=0)
    all_text_embeddings_tensor = torch.cat(all_text_embeddings_list, dim=0)
    
    all_image_embeddings_for_rdm = all_image_embeddings_tensor.to(device)
    img_pearson_corr, img_pearson_p, img_spearman_corr, img_spearman_p = \
        compute_rdm_and_correlation(all_image_embeddings_for_rdm, reference_rdm)

    image_results = {
        "image_rdm_pearson_correlation": img_pearson_corr,
        "image_rdm_pearson_p_value": img_pearson_p,
        "image_rdm_spearman_correlation": img_spearman_corr,
        "image_rdm_spearman_p_value": img_spearman_p,
        "image_embeddings": all_image_embeddings_tensor.numpy(),
        "image_names_from_loader": all_img_filenames_from_loader 
    }

    all_text_embeddings_for_rdm = all_text_embeddings_tensor.to(device)
    text_pearson_corr, text_pearson_p, text_spearman_corr, text_spearman_p = \
        compute_rdm_and_correlation(all_text_embeddings_for_rdm, reference_rdm)
        
    text_results = {
        "text_rdm_pearson_correlation": text_pearson_corr,
        "text_rdm_pearson_p_value": text_pearson_p,
        "text_rdm_spearman_correlation": text_spearman_corr,
        "text_rdm_spearman_p_value": text_spearman_p,
        "text_embeddings": all_text_embeddings_tensor.numpy() 
    }
    
    image_text_concat = torch.cat([all_image_embeddings_tensor, all_text_embeddings_tensor], dim=1)
    it_pearson_corr, it_pearson_p, it_spearman_corr, it_spearman_p = compute_rdm_and_correlation(image_text_concat, reference_rdm)
    image_text_concat_results = {
        "image_text_concat_pearson_correlation": it_pearson_corr,
        "image_text_concat_pearson_p_value": it_pearson_p,
        "image_text_concat_spearman_correlation": it_spearman_corr,
        "image_text_concat_spearman_p_value": it_spearman_p
    }
    
    final_results = {**image_results, **text_results, **image_text_concat_results}
    return final_results

def run_single_eval(
    epoch: int,
    ckpt_path: str,
    gpu_id: int,
    image_dir: str,
    rdm_path: str,
    output_dir: str,
    batch_size: int
) -> Dict:
    """Evaluate a single checkpoint on Things dataset (images and text concepts)."""
    logger.info(f"\n=== Evaluating epoch {epoch:>4d} on GPU {gpu_id} ===")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    reference_rdm = np.load(rdm_path)
    
    embedding_csv = os.path.join(image_dir, "hebart66_embedding_rescaled.csv")
    spose_embeddings = pd.read_csv(embedding_csv, index_col=0)
    # These are the image filenames, ordered according to the reference RDM.
    # Also used to derive text prompts.
    ordered_image_filenames = spose_embeddings.index.tolist() 
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # ThingsDataset uses these filenames to load images and derive prompts.
    dataset = ThingsDataset(image_dir, ordered_image_filenames, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Important for matching embedding order to RDM
        num_workers=4,
        pin_memory=True
        # Default collate_fn should work for (Tensor, str, str) items
    )
    
    model = load_clip_model_from_checkpoint(ckpt_path, gpu_id)
    if model is None:
        return {
            "epoch": epoch,
            "gpu_id": gpu_id,
            "error": "Failed to load model"
        }
    
    try:
        eval_results_dict = evaluate_things(
            model, 
            dataloader, 
            reference_rdm, 
            gpu_id
        )
        
        image_embeddings_to_save = eval_results_dict.pop("image_embeddings", None)
        image_names_from_loader = eval_results_dict.pop("image_names_from_loader", None)
        text_embeddings_to_save = eval_results_dict.pop("text_embeddings", None)

        if image_embeddings_to_save is not None:
            img_embed_path = os.path.join(output_dir, f"epoch_{epoch:03d}_gpu{gpu_id}_image_embeddings.npy")
            np.save(img_embed_path, image_embeddings_to_save)
            logger.info(f"[GPU {gpu_id}] Saved image embeddings to {img_embed_path}")

        if image_names_from_loader is not None: # These are from dataloader
            img_names_path = os.path.join(output_dir, f"epoch_{epoch:03d}_gpu{gpu_id}_image_names.txt")
            with open(img_names_path, 'w') as f:
                for name in image_names_from_loader:
                    f.write(f"{name}\n")
            logger.info(f"[GPU {gpu_id}] Saved image names (from loader) to {img_names_path}")
        
        if text_embeddings_to_save is not None:
            text_embed_path = os.path.join(output_dir, f"epoch_{epoch:03d}_gpu{gpu_id}_text_embeddings.npy")
            np.save(text_embed_path, text_embeddings_to_save)
            logger.info(f"[GPU {gpu_id}] Saved text embeddings to {text_embed_path}")
        
        output_for_csv = {
            "epoch": epoch,
            "gpu_id": gpu_id,
            "checkpoint": os.path.basename(ckpt_path),
            **eval_results_dict 
        }
        
        df = pd.DataFrame([output_for_csv])
        individual_path = os.path.join(output_dir, f"epoch_{epoch:03d}_gpu{gpu_id}.csv")
        df.to_csv(individual_path, index=False)
        
        logger.info(f"[GPU {gpu_id}] Epoch {epoch:>4d} Results:")
        logger.info(f"  Image Pearson R: {output_for_csv['image_rdm_pearson_correlation']:.4f} (p={output_for_csv['image_rdm_pearson_p_value']:.4e})")
        logger.info(f"  Image Spearman R: {output_for_csv['image_rdm_spearman_correlation']:.4f} (p={output_for_csv['image_rdm_spearman_p_value']:.4e})")
        logger.info(f"  Text Pearson R: {output_for_csv['text_rdm_pearson_correlation']:.4f} (p={output_for_csv['text_rdm_pearson_p_value']:.4e})")
        logger.info(f"  Text Spearman R: {output_for_csv['text_rdm_spearman_correlation']:.4f} (p={output_for_csv['text_rdm_spearman_p_value']:.4e})")
        logger.info(f"  Image-Text Concat Pearson R: {output_for_csv['image_text_concat_pearson_correlation']:.4f} (p={output_for_csv['image_text_concat_pearson_p_value']:.4e})")
        logger.info(f"  Image-Text Concat Spearman R: {output_for_csv['image_text_concat_spearman_correlation']:.4f} (p={output_for_csv['image_text_concat_spearman_p_value']:.4e})")
        return output_for_csv
        
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Error evaluating epoch {epoch}: {e}")
        return {
            "epoch": epoch,
            "gpu_id": gpu_id,
            "error": str(e)
        }
    finally:
        del model
        torch.cuda.empty_cache()

def gpu_worker(
    gpu_id: int,
    jobs: List[Tuple[int, str]],
    image_dir: str,
    rdm_path: str,
    output_dir: str,
    batch_size: int
) -> List[Dict]:
    """Worker process that sticks to one GPU and evaluates multiple checkpoints."""
    torch.cuda.set_device(gpu_id)
    results = []
    
    for epoch, ckpt_path in jobs:
        try:
            result = run_single_eval(
                epoch=epoch,
                ckpt_path=ckpt_path,
                gpu_id=gpu_id,
                image_dir=image_dir,
                rdm_path=rdm_path,
                output_dir=output_dir,
                batch_size=batch_size
            )
            results.append(result)
        except Exception as e:
            logger.error(f"[Worker-{gpu_id}] Epoch {epoch} failed: {e}")
            results.append({
                "epoch": epoch,
                "gpu_id": gpu_id,
                "error": str(e)
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Distributed evaluation of Things checkpoints")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=1000)
    parser.add_argument(
        "--num_gpus", 
        type=int, 
        default=0,
        help="Number of GPUs to use (0 = all available)"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/dev/shm/checkpoints/yfcc15m_litdata_20250506_013343_ddp_nodeNone/checkpoints",
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/huy28/ondemand/alberthu233/HebartImage1854",
        help="Directory containing Things images"
    )
    parser.add_argument(
        "--rdm_path",
        type=str,
        default="/home/huy28/ondemand/alberthu233/MMM_research_test/MMM_research/data/rdm.npy",
        help="Path to reference RDM numpy file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/things_evaluation_yfcc",
        help="Directory to save results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    logger.info(f"[START] Evaluating Things checkpoints {args.start_epoch} → {args.end_epoch}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Discover checkpoints
    ckpts = discover_checkpoints(args.checkpoints_dir, args.start_epoch, args.end_epoch)
    if not ckpts:
        logger.error("No checkpoints found in requested range.")
        return
    logger.info(f"Found {len(ckpts)} checkpoints: {[e for e, _ in ckpts]}")
    
    # GPU allocation
    num_avail = torch.cuda.device_count()
    if num_avail == 0:
        raise RuntimeError("No CUDA GPUs available")
    
    gpus_to_use = num_avail if args.num_gpus == 0 else min(args.num_gpus, num_avail)
    logger.info(f"Using {gpus_to_use} GPU(s)")
    
    # Distribute jobs across GPUs
    gpu_jobs = {i: [] for i in range(gpus_to_use)}
    for idx, job in enumerate(ckpts):
        gpu_jobs[idx % gpus_to_use].append(job)
    
    # Setup multiprocessing with explicit processes
    processes = []
    
    for gid, jobs in gpu_jobs.items():
        if not jobs:
            continue
            
        # Create non-daemonic process
        p = mp.Process(
            target=gpu_worker,
            args=(gid, jobs, args.image_dir, args.rdm_path, args.output_dir, args.batch_size),
            daemon=False
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Load and combine results from individual CSV files
    all_results = []
    for gid in gpu_jobs.keys():
        for epoch, _ in gpu_jobs[gid]:
            result_file = os.path.join(args.output_dir, f"epoch_{epoch:03d}_gpu{gid}.csv")
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                all_results.extend(df.to_dict('records'))
    
    # Create final summary
    if all_results:
        df = pd.DataFrame(all_results).sort_values("epoch")
        
        summary_path = os.path.join(
            args.output_dir,
            f"things_summary_{args.start_epoch}_{args.end_epoch}_gpus{gpus_to_use}.csv"
        )
        df.to_csv(summary_path, index=False)
        
        logger.info("\nEvaluation Summary:")
        
        img_pearson_corrs = df["image_rdm_pearson_correlation"].dropna()
        if len(img_pearson_corrs) > 0:
            logger.info(f"\nImage RDM Pearson Correlation Results:")
            logger.info(f"  Mean Correlation: {img_pearson_corrs.mean():.4f}")
            logger.info(f"  Max Correlation:  {img_pearson_corrs.max():.4f}")
            logger.info(f"  Valid Results: {len(img_pearson_corrs)}/{len(df)}")

        img_spearman_corrs = df["image_rdm_spearman_correlation"].dropna()
        if len(img_spearman_corrs) > 0:
            logger.info(f"\nImage RDM Spearman Correlation Results:")
            logger.info(f"  Mean Correlation: {img_spearman_corrs.mean():.4f}")
            logger.info(f"  Max Correlation:  {img_spearman_corrs.max():.4f}")
            logger.info(f"  Valid Results: {len(img_spearman_corrs)}/{len(df)}")

        text_pearson_corrs = df["text_rdm_pearson_correlation"].dropna()
        if len(text_pearson_corrs) > 0:
            logger.info(f"\nText RDM Pearson Correlation Results:")
            logger.info(f"  Mean Correlation: {text_pearson_corrs.mean():.4f}")
            logger.info(f"  Max Correlation:  {text_pearson_corrs.max():.4f}")
            logger.info(f"  Valid Results: {len(text_pearson_corrs)}/{len(df)}")

        text_spearman_corrs = df["text_rdm_spearman_correlation"].dropna()
        if len(text_spearman_corrs) > 0:
            logger.info(f"\nText RDM Spearman Correlation Results:")
            logger.info(f"  Mean Correlation: {text_spearman_corrs.mean():.4f}")
            logger.info(f"  Max Correlation:  {text_spearman_corrs.max():.4f}")
            logger.info(f"  Valid Results: {len(text_spearman_corrs)}/{len(df)}")
        
        logger.info(f"\n✅ Summary saved to {summary_path}")
    else:
        logger.error("\n⚠️ No successful evaluations recorded")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()