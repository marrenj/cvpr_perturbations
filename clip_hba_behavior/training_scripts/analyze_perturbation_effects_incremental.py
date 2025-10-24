"""
Incremental analyzer for perturbation sweep.
- Handles multiple perturbation lengths automatically
- Generates:
    1) Run-level alignment/loss plots
    2) Summary heatmap: perturb_epoch × perturb_length → final alignment / recovery ratio
- Tracks already-analyzed runs to avoid redundant processing
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Use a more flexible approach to find the output directory
OUTPUT_DIR = '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops'
CACHE_PATH = "analysis/analyzed_runs.json"
SUMMARY_CSV = "analysis/perturbation_summary.csv"
SUMMARY_HEATMAP = "analysis/perturbation_heatmap.png"
FIGURE_DIR = "analysis/run_plots"
PERMANENT_HARM_THRESHOLD = 0.95  # proportion of baseline alignment
BASELINE_PATH = "output/baseline/metrics.csv"

os.makedirs(FIGURE_DIR, exist_ok=True)

# --- Load baseline ---
def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        print("⚠️ Baseline metrics not found. Skipping baseline comparison.")
        return None
    df = pd.read_csv(BASELINE_PATH)
    df["epoch"] = df["epoch"].astype(int)
    return df.set_index("epoch")

baseline_df = load_baseline()

# --- Load cache ---
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_cache(processed):
    with open(CACHE_PATH, "w") as f:
        json.dump(list(processed), f)

processed = load_cache()

# --- Identify new runs ---
def list_new_runs(processed):
    # Look for metrics.csv files in the output directories
    all_runs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "metrics.csv")))
    return [r for r in all_runs if r not in processed and os.path.getsize(r) > 500]

new_runs = list_new_runs(processed)
if not new_runs:
    print("No new runs to analyze.")
    exit()

# --- Analyze a single run ---
def analyze_run(path, baseline_df):
    run_id = os.path.basename(os.path.dirname(path))
    # Parse perturb_epoch and perturb_length from run_id
    parts = run_id.split("_")
    try:
        perturb_epoch = int(parts[2][1:])  # e.g., e2
        perturb_length = int(parts[3][1:])  # e.g., l4
    except:
        perturb_epoch, perturb_length = None, None

    df = pd.read_csv(path)
    if "epoch" not in df.columns or "alignment" not in df.columns:
        print(f"Skipping {run_id}: missing required columns.")
        return None
    df["epoch"] = df["epoch"].astype(int)
    last_epoch = df["epoch"].max()
    final_alignment = df["alignment"].iloc[-1]
    final_loss = df["loss"].iloc[-1] if "loss" in df.columns else np.nan

    permanently_harmed = False
    recovery_ratio = np.nan
    if baseline_df is not None and last_epoch in baseline_df.index:
        baseline_final = baseline_df.loc[last_epoch, "alignment"]
        recovery_ratio = final_alignment / baseline_final
        permanently_harmed = recovery_ratio < PERMANENT_HARM_THRESHOLD

    # --- Generate run-level plots ---
    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["alignment"], label="Run", color="red")
    if baseline_df is not None:
        plt.plot(baseline_df.index, baseline_df["alignment"], label="Baseline", color="black", lw=2)
    plt.title(f"Alignment: {run_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"{run_id}_alignment.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["loss"], label="Run", color="blue")
    if baseline_df is not None:
        plt.plot(baseline_df.index, baseline_df["loss"], label="Baseline", color="black", lw=2)
    plt.title(f"Loss: {run_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f"{run_id}_loss.png"))
    plt.close()

    return {
        "run_id": run_id,
        "perturb_epoch": perturb_epoch,
        "perturb_length": perturb_length,
        "final_alignment": final_alignment,
        "final_loss": final_loss,
        "recovery_ratio": recovery_ratio,
        "permanently_harmed": permanently_harmed,
        "epochs": last_epoch,
        "metrics_path": path
    }

# --- Process all new runs ---
results = []
for path in new_runs:
    result = analyze_run(path, baseline_df)
    if result:
        results.append(result)
        processed.add(path)

# --- Update summary CSV ---
if os.path.exists(SUMMARY_CSV):
    summary_df = pd.read_csv(SUMMARY_CSV)
    summary_df = pd.concat([summary_df, pd.DataFrame(results)], ignore_index=True)
else:
    summary_df = pd.DataFrame(results)
summary_df.to_csv(SUMMARY_CSV, index=False)
save_cache(processed)

# --- Generate summary heatmap ---
pivot = summary_df.pivot_table(
    index="perturb_epoch",
    columns="perturb_length",
    values="recovery_ratio"
)
plt.figure(figsize=(8, 6))
plt.imshow(pivot, origin="lower", aspect="auto", cmap="coolwarm", vmin=0, vmax=1.1)
plt.colorbar(label="Recovery Ratio")
plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.xlabel("Perturbation Length")
plt.ylabel("Perturbation Start Epoch")
plt.title("Perturbation Sweep Heatmap (Recovery Ratio)")
plt.tight_layout()
plt.savefig(SUMMARY_HEATMAP, dpi=150)
plt.close()

print(f"✅ Analyzed {len(new_runs)} new runs. Updated summary CSV and heatmap.")
