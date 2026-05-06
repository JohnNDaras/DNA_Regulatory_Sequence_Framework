import matplotlib.pyplot as plt

def plot_history(history, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history.get("train_accs", []), label="train_acc")
    plt.plot(history.get("val_accs", []), label="val_acc")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)

def plot_metric_bar(df, metric_col="val_auprc", label_col="model", title=None, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.bar(df[label_col].astype(str), df[metric_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric_col)
    plt.title(title or metric_col)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
