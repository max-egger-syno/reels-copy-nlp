import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT, GRAY = "#0f6b6b", "#9aa3b2"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


MODEL_LABELS = {"random": "Random", "length_only": "Opening pace + length", "devices_anew_perplexity_lr": "Devices+ANEW+PPL",
                "tfidf_lr": "TF-IDF (pooled)", "tfidf_lr_opening_only": "TF-IDF, opening only",
                "tfidf_lr_per_creator": "TF-IDF (per creator)", "bertimbau_frozen_lr": "BERT (BERTimbau), frozen"}


def fig_e3(e3: pd.DataFrame, path, human=0.613, tan=0.656, people=None):
    d = e3.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(3.5, 0.32 * len(d) + 0.9))
    y = np.arange(len(d))
    ax.barh(y, d["accuracy"], color=[ACCENT if m != "random" else GRAY for m in d["model"]], height=0.55)
    ax.errorbar(d["accuracy"], y, xerr=[d["accuracy"] - d["ci_low"], d["ci_high"] - d["accuracy"]],
                fmt="none", ecolor="#1b2230", capsize=2, lw=0.8)
    refs = [(human, "Tan humans", "--"), (tan, "Tan model", "-.")]
    if people is not None:
        refs.append((people, "our humans", ":"))
    for x, lab, ls in refs:
        ax.axvline(x, color="#5b6475", ls=ls, lw=0.8, label=f"{lab} ({x:.2f})")
    ax.axvline(0.5, color="#1b2230", lw=0.6)
    ax.set_yticks(y, [MODEL_LABELS.get(m, m) for m in d["model"]], fontsize=7)
    ax.set_xlim(0.3, 0.8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Pairwise accuracy, newest reels (95% CI)", fontsize=7)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_learning_curve(lc: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for (model, metric), color, ls in ((("tfidf_lr", "test_pair_accuracy"), ACCENT, "-"),
                                       (("tfidf_lr", "train_accuracy"), ACCENT, ":"),
                                       (("bertimbau_frozen_lr", "test_pair_accuracy"), "#9a5b00", "-"),
                                       (("bertimbau_frozen_lr", "train_accuracy"), "#9a5b00", ":")):
        g = lc[lc["model"] == model]
        if g.empty:
            continue
        s = g.groupby("n_train_reels")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(s["n_train_reels"], s["mean"], color=color, ls=ls, label=f"{model} ({metric.split('_')[0]})")
        ax.fill_between(s["n_train_reels"], s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.12)
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel("Training reels")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_e4(matrix: pd.DataFrame, creators: pd.DataFrame, path):
    order = creators.sort_values(["sector", "username"])["username"].tolist()
    order = [u for u in order if u in set(matrix["train_creator"])]
    m = matrix.pivot(index="train_creator", columns="test_creator", values="balanced_accuracy").reindex(
        index=order, columns=order)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(m.to_numpy(), cmap="BuGn", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_xlabel("Tested on creator")
    ax.set_ylabel("Trained on creator")
    fig.colorbar(im, ax=ax, label="Balanced accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_e1(e1: pd.DataFrame, path, per_sector: int = 6):
    if e1.empty:
        return
    sectors = sorted(e1["sector"].dropna().unique())
    fig, axes = plt.subplots(1, len(sectors), figsize=(2.6 * len(sectors), 3.4), squeeze=False)
    for ax, sector in zip(axes[0], sectors):
        g = e1[(e1["sector"] == sector) & (e1["label"] == "y_viral")]
        g = g.groupby(["term", "direction"], as_index=False).agg(p=("p_viral", "mean"), s=("stability", "mean"))
        top = pd.concat([g[g.direction == "viral"].nlargest(per_sector, "p"),
                         g[g.direction == "flop"].nsmallest(per_sector, "p")])
        top = top.sort_values("p")
        ax.barh(range(len(top)), top["p"] - 0.5, left=0.5,
                color=[ACCENT if d == "viral" else GRAY for d in top["direction"]])
        ax.set_yticks(range(len(top)), top["term"], fontsize=7)
        ax.axvline(0.5, color="#1b2230", lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_title(sector, fontsize=8)
        ax.set_xlabel("P(viral | word)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
