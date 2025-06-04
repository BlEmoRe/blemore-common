import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("validation_summary.csv")

# Compute mean accuracy per encoder and model
grouped = df.groupby(["encoder", "model"])[["best_acc_presence", "best_acc_salience"]].mean().reset_index()

# Melt into long format
melted = grouped.melt(id_vars=["encoder", "model"],
                      value_vars=["best_acc_presence", "best_acc_salience"],
                      var_name="metric", value_name="accuracy")

# One plot per metric (presence/salience), comparing encoders & models
for metric in ["best_acc_presence", "best_acc_salience"]:
    plt.figure(figsize=(10, 6))
    data = melted[melted["metric"] == metric]
    sns.barplot(data=data, x="encoder", y="accuracy", hue="model", palette="Set2")
    plt.title(f"Mean {metric.replace('_', ' ').title()} by Encoder and Model")
    # plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Encoder")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()



df = pd.read_csv("validation_summary_subsampled.csv")

# Compute mean accuracy per encoder and model
grouped = df.groupby(["encoder", "model"])[["best_acc_presence", "best_acc_salience"]].mean().reset_index()

# Melt into long format
melted = grouped.melt(id_vars=["encoder", "model"],
                      value_vars=["best_acc_presence", "best_acc_salience"],
                      var_name="metric", value_name="accuracy")

# One plot per metric (presence/salience), comparing encoders & models
for metric in ["best_acc_presence", "best_acc_salience"]:
    plt.figure(figsize=(10, 6))
    data = melted[melted["metric"] == metric]
    sns.barplot(data=data, x="encoder", y="accuracy", hue="model", palette="Set2")
    plt.title(f"Mean {metric.replace('_', ' ').title()} by Encoder and Model")
    # plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Encoder")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

