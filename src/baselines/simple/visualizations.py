import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_grid_heatmap(grid, metric_index, title, cmap="viridis"):
    """
    Plots a heatmap from the alpha-beta grid.

    Args:
        grid (list of tuples): Each tuple contains (alpha, beta, presence_acc, salience_acc)
        metric_index (int): 2 for presence acc, 3 for salience acc
        title (str): Plot title
        cmap (str): Colormap
    """
    alphas = sorted(set(x[0] for x in grid))
    betas = sorted(set(x[1] for x in grid))

    alpha_to_idx = {a: i for i, a in enumerate(alphas)}
    beta_to_idx = {b: i for i, b in enumerate(betas)}

    heatmap = np.full((len(betas), len(alphas)), np.nan)

    for a, b, pres, sal in grid:
        value = pres if metric_index == 2 else sal
        heatmap[beta_to_idx[b], alpha_to_idx[a]] = value

    plt.figure(figsize=(10, 6))
    im = plt.imshow(heatmap, origin='lower', aspect='auto', cmap=cmap,
                    extent=[min(alphas), max(alphas), min(betas), max(betas)])
    plt.colorbar(im)
    plt.xlabel("Alpha (presence threshold)")
    plt.ylabel("Beta (salience threshold)")
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def summarize_prediction_distribution(label_dict):
    blend_types = []

    for preds in label_dict.values():
        if len(preds) == 1:
            blend_types.append("single")
        elif len(preds) == 2:
            saliences = sorted([round(p["salience"]) for p in preds], reverse=True)
            if saliences == [70, 30]:
                blend_types.append("70/30")
            elif saliences == [50, 50]:
                blend_types.append("50/50")
            else:
                blend_types.append("other_blend")
        else:
            blend_types.append("invalid")

    counter = Counter(blend_types)

    # Print counts
    for k, v in counter.items():
        print(f"{k}: {v}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(counter.keys(), counter.values(), color='skyblue')
    plt.title("Distribution of Predicted Emotion Types")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()