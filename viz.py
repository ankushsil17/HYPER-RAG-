import os
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# Simple line plot: top-k vs F1 and hallucinations (1-groundedness)


def plot_topk_curves(df, plots_dir):
    ensure_dir(plots_dir)
    for dname in df["dataset"].unique():
        sub = df[(df.dataset==dname) & (df.param=="top_k")]
        if sub.empty: continue
        pivot_f1 = sub.pivot_table(index="value", values="f1", aggfunc="mean").reset_index()
        pivot_h = sub.pivot_table(index="value", values="groundedness", aggfunc="mean").reset_index()
        plt.figure(figsize=(6,4))
        plt.plot(pivot_f1["value"], pivot_f1["f1"], marker="o", label="F1")
        plt.plot(pivot_h["value"], 1.0-pivot_h["groundedness"], marker="s", label="Hallucination rate")
        plt.xlabel("top-k"); plt.title(f"{dname}: top-k sweep")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dname}_topk.png"))
        plt.close()
