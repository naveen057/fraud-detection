"""Stage 2 – exploratory data analysis (figures -> reports/figures, insights -> reports/eda_summary.json)."""
import json

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fraud.config import FIG_DIR, NUMERIC, REPORTS_DIR, TARGET
from fraud.data import load_processed
from fraud.features import add_derived

sns.set_theme(style="whitegrid", context="notebook")
PAL = {0: "#4C72B0", 1: "#C44E52"}


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=130); plt.close(fig)


def main():
    df = add_derived(load_processed())
    fraud, legit = df[df[TARGET] == 1], df[df[TARGET] == 0]
    s = {}

    # 1. class balance --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax[0].bar(["Legit", "Fraud"], counts.values, color=[PAL[0], PAL[1]])
    ax[0].set_title("Class counts (linear)"); ax[0].set_ylabel("transactions")
    ax[1].bar(["Legit", "Fraud"], counts.values, color=[PAL[0], PAL[1]]); ax[1].set_yscale("log")
    ax[1].set_title("Class counts (log scale)")
    for a in ax:
        for i, v in enumerate(counts.values):
            a.text(i, v, f"{v:,}", ha="center", va="bottom")
    save(fig, "eda_01_class_balance.png")
    s["fraud_rate_pct"] = round(100 * df[TARGET].mean(), 3)

    # 2. amount ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, g in df.groupby(TARGET):
        sns.kdeplot(np.log10(g["amount"]), ax=ax[0], fill=True, alpha=.35, color=PAL[k],
                    label="Fraud" if k else "Legit", common_norm=False)
        sns.kdeplot(np.log10(g["amount_to_avg_ratio"]), ax=ax[1], fill=True, alpha=.35,
                    color=PAL[k], label="Fraud" if k else "Legit", common_norm=False)
    ax[0].set_xlabel("log10(amount)"); ax[0].set_title("Amount distribution")
    ax[1].set_xlabel("log10(amount / user 30-day average)"); ax[1].set_title("Spend relative to own history")
    ax[0].legend(); save(fig, "eda_02_amount.png")
    s["median_amount"] = {"legit": round(float(legit.amount.median()), 2), "fraud": round(float(fraud.amount.median()), 2)}

    # 3. time & category ------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    by_hour = df.groupby("hour")[TARGET].mean() * 100
    ax[0].bar(by_hour.index, by_hour.values, color="#C44E52"); ax[0].set_xlabel("hour of day")
    ax[0].set_ylabel("fraud rate (%)"); ax[0].set_title("Fraud rate by hour")
    by_cat = (df.groupby("merchant_category")[TARGET].mean() * 100).sort_values()
    ax[1].barh(by_cat.index, by_cat.values, color="#C44E52"); ax[1].set_xlabel("fraud rate (%)")
    ax[1].set_title("Fraud rate by merchant category")
    save(fig, "eda_03_hour_category.png")
    s["fraud_rate_by_category_pct"] = by_cat.round(2).to_dict()
    s["night_fraud_rate_pct"] = round(100 * df[df.is_night == 1][TARGET].mean(), 2)
    s["day_fraud_rate_pct"] = round(100 * df[df.is_night == 0][TARGET].mean(), 2)

    # 4. behavioural / velocity signals ---------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, title in zip(ax, ["is_foreign", "device_new", "channel"],
                             ["Foreign transaction", "New device", "Channel"]):
        r = df.groupby(col)[TARGET].mean() * 100
        a.bar(r.index.astype(str), r.values, color="#C44E52"); a.set_title(f"Fraud rate: {title}")
        a.set_ylabel("fraud rate (%)")
    save(fig, "eda_04_behaviour_flags.png")
    v = df.groupby(df["txn_count_1h"].clip(upper=5))[TARGET].mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(v.index.astype(str), v.values, color="#C44E52")
    ax[0].set_xlabel("transactions by same user in prior hour (5 = 5+)"); ax[0].set_ylabel("fraud rate (%)")
    ax[0].set_title("Velocity: card-testing bursts")
    sns.boxplot(data=df, x=TARGET, y=np.log10(df["distance_from_home_km"] + 1), ax=ax[1],
                palette=PAL, hue=TARGET, legend=False)
    ax[1].set_xticks([0, 1], ["Legit", "Fraud"]); ax[1].set_ylabel("log10(distance from home km + 1)")
    ax[1].set_title("Distance from home")
    save(fig, "eda_05_velocity_distance.png")
    s["fraud_rate_by_txn_count_1h_pct"] = v.round(2).to_dict()
    s["fraud_rate_foreign_pct"] = {int(k): round(float(x), 2) for k, x in (df.groupby("is_foreign")[TARGET].mean() * 100).items()}
    s["fraud_rate_new_device_pct"] = {int(k): round(float(x), 2) for k, x in (df.groupby("device_new")[TARGET].mean() * 100).items()}

    # 5. correlation ----------------------------------------------------------
    corr = df[NUMERIC + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Feature correlation matrix"); save(fig, "eda_06_correlation.png")
    s["top_correlations_with_fraud"] = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(6).round(3).to_dict()

    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
