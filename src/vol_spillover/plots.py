import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vol_spillover.plots")


def generate_plots(
    dy_res: dict,
    tvp_res: dict,
    q_res: dict,
    freq_res: dict,
    gfevd_static: pd.DataFrame,
    config: dict
):
    reports_dir = config.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. TCI Overlay Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    if dy_res and "tci" in dy_res:
        ax.plot(dy_res["tci"].index, dy_res["tci"], label="Rolling DY", color="blue", alpha=0.8)
    if tvp_res and "tci" in tvp_res:
        ax.plot(tvp_res["tci"].index, tvp_res["tci"], label="TVP-VAR", color="red", linestyle="--", alpha=0.8)
    if q_res and 0.5 in q_res and "tci" in q_res[0.5]:
        ax.plot(q_res[0.5]["tci"].index, q_res[0.5]["tci"], label="Quantile 0.50", color="green", alpha=0.7)
    if q_res and 0.95 in q_res and "tci" in q_res[0.95]:
        ax.plot(q_res[0.95]["tci"].index, q_res[0.95]["tci"], label="Quantile 0.95 (Tail)", color="orange", linestyle=":", alpha=0.9)

    ax.set_title("Total Connectedness Index (TCI) Over Time", fontsize=14, fontweight="bold")
    ax.set_ylabel("TCI (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, "tci_overlay.png"), dpi=300)
    plt.close(fig)

    # 2. NET Small Multiples
    if dy_res and "net" in dy_res:
        net_df = dy_res["net"]
        n_assets = len(net_df.columns)
        n_cols = 3
        n_rows = int(np.ceil(n_assets / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten() if n_assets > 1 else [axes]

        for i, col in enumerate(net_df.columns):
            ax = axes[i]
            s = net_df[col]
            ax.plot(s.index, s, color="teal", lw=1.2)
            ax.axhline(0, color="black", linestyle="--", lw=0.8)
            ax.fill_between(s.index, s, 0, where=(s >= 0), color="green", alpha=0.3)
            ax.fill_between(s.index, s, 0, where=(s < 0), color="red", alpha=0.3)
            ax.set_title(col, fontsize=10, fontweight="bold")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Net Volatility Spillover per Asset (NET_i %)", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(reports_dir, "net_small_multiples.png"), dpi=300)
        plt.close(fig)

    # 3. NPDC Candidate toward Receptor Indices
    instruments = config.get("instruments", [])
    index_ids = [inst["id"] for inst in instruments if inst.get("role") == "receptor"]
    candidate_ids = [inst["id"] for inst in instruments if inst.get("role") == "candidato"]

    if dy_res and "npdc_raw" in dy_res:
        dates = dy_res["dates"]
        npdc_raw = dy_res["npdc_raw"]

        for idx_target in index_ids:
            fig, ax = plt.subplots(figsize=(10, 5))
            for cand in candidate_ids:
                key = (cand, idx_target)
                if key in npdc_raw:
                    vals = npdc_raw[key]
                    ax.plot(dates[:len(vals)], vals, label=f"{cand} -> {idx_target}")

            ax.axhline(0, color="black", linestyle="--", lw=0.8)
            ax.set_title(f"Net Pairwise Directional Connectedness (NPDC) towards {idx_target.upper()}", fontsize=12, fontweight="bold")
            ax.set_ylabel("NPDC (%)")
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(reports_dir, f"npdc_{idx_target}.png"), dpi=300)
            plt.close(fig)

    # 4. Connectedness Heatmaps (Full Sample & Last 60 Days)
    if gfevd_static is not None and not gfevd_static.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(gfevd_static.values, cmap="YlOrRd")
        ax.set_xticks(range(len(gfevd_static.columns)))
        ax.set_yticks(range(len(gfevd_static.index)))
        ax.set_xticklabels(gfevd_static.columns, rotation=45, ha="right")
        ax.set_yticklabels(gfevd_static.index)

        for i in range(len(gfevd_static.index)):
            for j in range(len(gfevd_static.columns)):
                ax.text(j, i, f"{gfevd_static.iloc[i, j]*100:.1f}", ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax, label="GFEVD (%)")
        ax.set_title("Full Sample Volatility Spillover Matrix (%)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, "heatmap_full_sample.png"), dpi=300)
        plt.close(fig)

    # 5. Frequency Heatmap
    if freq_res:
        bands = list(freq_res.keys())
        fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4))
        if len(bands) == 1:
            axes = [axes]

        for b_idx, band_name in enumerate(bands):
            ax = axes[b_idx]
            g_mat = freq_res[band_name]["gfevd"]
            im = ax.imshow(g_mat.values, cmap="Blues")
            ax.set_xticks(range(len(g_mat.columns)))
            ax.set_yticks(range(len(g_mat.index)))
            ax.set_xticklabels(g_mat.columns, rotation=45, ha="right")
            ax.set_yticklabels(g_mat.index)
            ax.set_title(f"Band: {band_name.upper()}", fontsize=11, fontweight="bold")

        fig.suptitle("Frequency Connectedness across Bands", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(reports_dir, "frequency_heatmap.png"), dpi=300)
        plt.close(fig)

    logger.info(f"Plots successfully generated and saved in {reports_dir}")
