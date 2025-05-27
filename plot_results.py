import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("validation_summary.csv")

linear_df = df[df["model"] == "linear"]

plt.figure(figsize=(12, 6))
sns.lineplot(data=linear_df, x="fold", y="acc_presence", hue="encoder", marker="o")
plt.title("Accuracy Presence by Fold – Linear Models")
plt.ylabel("Accuracy Presence")
plt.xlabel("Fold")
plt.grid(True)
plt.legend(title="Encoder")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=linear_df, x="fold", y="acc_salience", hue="encoder", marker="o")
plt.title("Accuracy Salience by Fold – Linear Models")
plt.ylabel("Accuracy Salience")
plt.xlabel("Fold")
plt.grid(True)
plt.legend(title="Encoder")
plt.tight_layout()
plt.show()

rnn_df = df[df["model"] == "rnn"]

plt.figure(figsize=(12, 6))
sns.lineplot(data=rnn_df, x="fold", y="acc_presence", hue="encoder", marker="o")
plt.title("Accuracy Presence by Fold – RNN Models")
plt.ylabel("Accuracy Presence")
plt.xlabel("Fold")
plt.grid(True)
plt.legend(title="Encoder")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=rnn_df, x="fold", y="acc_salience", hue="encoder", marker="o")
plt.title("Accuracy Salience by Fold – RNN Models")
plt.ylabel("Accuracy Salience")
plt.xlabel("Fold")
plt.grid(True)
plt.legend(title="Encoder")
plt.tight_layout()
plt.show()

