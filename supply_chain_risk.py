"""
Supply Chain Disruption Risk Tracker
======================================
Author  : Anushka Raj
GitHub  : github.com/anushka747

Objective
---------
Analyse supplier performance across lead times, delivery delays, stockout events,
and defect rates to quantify disruption risk, surface at-risk supply nodes, and
recommend targeted mitigation actions before disruptions hit production.

Sections
--------
1. Data Generation & Overview
2. Data Cleaning & Feature Engineering
3. Exploratory Data Analysis
4. Anomaly Detection — Lead Time Spikes
5. Supplier Risk Scoring (Composite Index)
6. Financial Impact Quantification
7. Time-Series Trend Analysis
8. SQL-Style Aggregations
9. Key Insights & Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ── Colour palette ────────────────────────────────────────────────────────────
RED    = "#D85A30"
AMBER  = "#BA7517"
GREEN  = "#1D9E75"
BLUE   = "#185FA5"
GRAY   = "#888780"
DARK   = "#2D2D2D"

RISK_COLORS = {"Critical": RED, "High": AMBER, "Medium": BLUE, "Low": GREEN}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(seed=42):
    rng = np.random.default_rng(seed)

    suppliers = {
        "SUP-001": {"name": "AlphaComponents",  "region": "China",         "category": "Electronics",  "base_lt": 14, "reliability": 0.62},
        "SUP-002": {"name": "BetaMetals",        "region": "Germany",       "category": "Raw Materials","base_lt": 7,  "reliability": 0.91},
        "SUP-003": {"name": "GammaParts",        "region": "India",         "category": "Mechanical",  "base_lt": 10, "reliability": 0.78},
        "SUP-004": {"name": "DeltaLogistics",    "region": "USA",           "category": "Packaging",   "base_lt": 5,  "reliability": 0.85},
        "SUP-005": {"name": "EpsilonTextiles",   "region": "Bangladesh",    "category": "Textiles",    "base_lt": 21, "reliability": 0.55},
        "SUP-006": {"name": "ZetaElectronics",   "region": "Taiwan",        "category": "Electronics", "base_lt": 12, "reliability": 0.88},
        "SUP-007": {"name": "EtaChemicals",      "region": "China",         "category": "Chemicals",   "base_lt": 9,  "reliability": 0.67},
        "SUP-008": {"name": "ThetaPlastics",     "region": "Vietnam",       "category": "Raw Materials","base_lt": 8,  "reliability": 0.80},
    }

    records = []
    dates   = pd.date_range("2022-01-01", "2024-12-31", freq="W")

    for sid, meta in suppliers.items():
        for date in dates:
            # Inject seasonal disruptions (Q3 = peak disruption for Asia suppliers)
            seasonal_factor = 1.0
            if meta["region"] in ["China", "Bangladesh", "Vietnam"] and date.month in [7, 8, 9]:
                seasonal_factor = 1.35

            # Random shock events (port strikes, raw material shortages)
            shock = 1.0
            if rng.random() < (1 - meta["reliability"]) * 0.15:
                shock = rng.uniform(1.5, 3.2)

            lead_time      = max(1, meta["base_lt"] * seasonal_factor * shock + rng.normal(0, 2))
            promised_lt    = meta["base_lt"] * 1.1
            delay_days     = max(0, lead_time - promised_lt)
            on_time        = delay_days == 0
            defect_rate    = max(0, (1 - meta["reliability"]) * rng.uniform(0.5, 1.5) * shock)
            stockout       = (delay_days > meta["base_lt"] * 0.5) and rng.random() < 0.4
            order_value    = rng.uniform(5000, 80000)
            disruption_cost = (delay_days * order_value * 0.003) + (stockout * order_value * 0.12)

            records.append({
                "date":            date,
                "supplier_id":     sid,
                "supplier_name":   meta["name"],
                "region":          meta["region"],
                "category":        meta["category"],
                "lead_time_days":  round(lead_time, 1),
                "promised_lt":     round(promised_lt, 1),
                "delay_days":      round(delay_days, 1),
                "on_time_delivery":on_time,
                "defect_rate_pct": round(defect_rate * 100, 2),
                "stockout_event":  stockout,
                "order_value_usd": round(order_value, 2),
                "disruption_cost_usd": round(disruption_cost, 2),
            })

    return pd.DataFrame(records)


df = generate_data()

print("=" * 65)
print("SUPPLY CHAIN DISRUPTION RISK TRACKER")
print("=" * 65)
print(f"  Records          : {len(df):,}")
print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Suppliers tracked: {df['supplier_id'].nunique()}")
print(f"  On-time rate     : {df['on_time_delivery'].mean():.1%}")
print(f"  Avg delay        : {df['delay_days'].mean():.1f} days")
print(f"  Stockout events  : {df['stockout_event'].sum():,}")
print(f"  Total disruption cost: ${df['disruption_cost_usd'].sum():,.0f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

df["year_month"]   = df["date"].dt.to_period("M")
df["quarter"]      = df["date"].dt.to_period("Q").astype(str)
df["lead_time_vs_promised"] = df["lead_time_days"] - df["promised_lt"]

# Rolling 4-week average lead time per supplier
df = df.sort_values(["supplier_id", "date"])
df["rolling_lt_4w"] = (
    df.groupby("supplier_id")["lead_time_days"]
    .transform(lambda x: x.rolling(4, min_periods=1).mean())
)

# Anomaly flag: lead time > mean + 2 std per supplier
stats = df.groupby("supplier_id")["lead_time_days"].agg(["mean", "std"]).reset_index()
stats.columns = ["supplier_id", "lt_mean", "lt_std"]
df = df.merge(stats, on="supplier_id")
df["anomaly"] = df["lead_time_days"] > (df["lt_mean"] + 2 * df["lt_std"])

print(f"  Anomalies detected (lead time > mean+2σ): {df['anomaly'].sum():,} events")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUPPLIER RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

risk_df = df.groupby(["supplier_id", "supplier_name", "region", "category"]).agg(
    total_orders        = ("order_value_usd",     "count"),
    on_time_rate        = ("on_time_delivery",     "mean"),
    avg_delay_days      = ("delay_days",           "mean"),
    avg_defect_rate     = ("defect_rate_pct",      "mean"),
    stockout_rate       = ("stockout_event",       "mean"),
    total_disruption_cost=("disruption_cost_usd",  "sum"),
    anomaly_rate        = ("anomaly",              "mean"),
    avg_order_value     = ("order_value_usd",      "mean"),
).reset_index()

# Composite risk score (0–100, higher = riskier)
# Weights chosen to reflect business impact:
# delay=30%, defect=25%, stockout=25%, anomaly=20%
risk_df["delay_score"]   = (risk_df["avg_delay_days"]   / risk_df["avg_delay_days"].max())   * 30
risk_df["defect_score"]  = (risk_df["avg_defect_rate"]  / risk_df["avg_defect_rate"].max())  * 25
risk_df["stockout_score"]= (risk_df["stockout_rate"]    / risk_df["stockout_rate"].max())    * 25
risk_df["anomaly_score"] = (risk_df["anomaly_rate"]     / risk_df["anomaly_rate"].max())     * 20
risk_df["risk_score"]    = (risk_df["delay_score"] + risk_df["defect_score"] +
                            risk_df["stockout_score"] + risk_df["anomaly_score"]).round(1)

def risk_tier(score):
    if score >= 60: return "Critical"
    if score >= 40: return "High"
    if score >= 20: return "Medium"
    return "Low"

risk_df["risk_tier"] = risk_df["risk_score"].apply(risk_tier)
risk_df = risk_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

print("=" * 65)
print("SUPPLIER RISK SCORECARD")
print("=" * 65)
cols = ["supplier_name", "region", "risk_score", "risk_tier",
        "on_time_rate", "avg_delay_days", "avg_defect_rate", "total_disruption_cost"]
display = risk_df[cols].copy()
display["on_time_rate"]          = display["on_time_rate"].map("{:.1%}".format)
display["avg_delay_days"]        = display["avg_delay_days"].map("{:.1f}d".format)
display["avg_defect_rate"]       = display["avg_defect_rate"].map("{:.2f}%".format)
display["total_disruption_cost"] = display["total_disruption_cost"].map("${:,.0f}".format)
print(display.to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. CHART 1 — EDA OVERVIEW (2×3 grid)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Supply Chain Risk — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

# 4a. On-time delivery rate by supplier
ax = axes[0, 0]
otr = risk_df.sort_values("on_time_rate")
colors = [RISK_COLORS[t] for t in otr["risk_tier"]]
bars = ax.barh(otr["supplier_name"], otr["on_time_rate"] * 100,
               color=colors, edgecolor="white", height=0.6)
ax.axvline(df["on_time_delivery"].mean() * 100, color=DARK, linestyle="--", lw=1.5,
           label=f"Avg {df['on_time_delivery'].mean():.1%}")
for bar, val in zip(bars, otr["on_time_rate"]):
    ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=9)
ax.set_title("On-Time Delivery Rate by Supplier")
ax.set_xlabel("On-Time Rate (%)")
ax.set_xlim(0, 115)
ax.legend(fontsize=9)
patches = [mpatches.Patch(color=v, label=k) for k, v in RISK_COLORS.items()]
ax.legend(handles=patches, fontsize=8, loc="lower right")

# 4b. Avg delay days by region
ax = axes[0, 1]
reg = df.groupby("region")["delay_days"].mean().sort_values(ascending=False)
bar_colors = [RED if v > 3 else AMBER if v > 1.5 else GREEN for v in reg.values]
bars = ax.bar(reg.index, reg.values, color=bar_colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, reg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.1f}d", ha="center", va="bottom", fontsize=9)
ax.set_title("Average Delay Days by Region")
ax.set_ylabel("Avg Delay (days)")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

# 4c. Monthly disruption cost trend
ax = axes[0, 2]
monthly = df.groupby("year_month")["disruption_cost_usd"].sum().reset_index()
monthly["year_month_str"] = monthly["year_month"].astype(str)
monthly_q = monthly.iloc[::4]  # quarterly labels
ax.plot(range(len(monthly)), monthly["disruption_cost_usd"] / 1000,
        color=RED, linewidth=2, alpha=0.8)
ax.fill_between(range(len(monthly)), monthly["disruption_cost_usd"] / 1000,
                alpha=0.12, color=RED)
ax.set_xticks(range(0, len(monthly), 4))
ax.set_xticklabels([monthly["year_month_str"].iloc[i] for i in range(0, len(monthly), 4)],
                   rotation=30, ha="right", fontsize=8)
ax.set_title("Monthly Disruption Cost Trend")
ax.set_ylabel("Cost ($K)")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.0f}K"))

# 4d. Defect rate vs delay scatter
ax = axes[1, 0]
for _, row in risk_df.iterrows():
    ax.scatter(row["avg_delay_days"], row["avg_defect_rate"],
               s=row["avg_order_value"] / 500,
               color=RISK_COLORS[row["risk_tier"]], alpha=0.85, edgecolors="white", lw=0.8)
    ax.annotate(row["supplier_name"].replace("Components","").replace("Electronics","")
                .replace("Metals","").replace("Parts","").replace("Logistics","")
                .replace("Textiles","").replace("Chemicals","").replace("Plastics",""),
                (row["avg_delay_days"], row["avg_defect_rate"]),
                fontsize=8, ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points")
ax.set_title("Delay vs Defect Rate (bubble = order value)")
ax.set_xlabel("Avg Delay (days)")
ax.set_ylabel("Avg Defect Rate (%)")
patches = [mpatches.Patch(color=v, label=k) for k, v in RISK_COLORS.items()]
ax.legend(handles=patches, fontsize=8)

# 4e. Stockout events by quarter
ax = axes[1, 1]
qtr_stockout = df.groupby("quarter")["stockout_event"].sum()
bar_colors_q = [RED if v > qtr_stockout.mean() * 1.2 else AMBER
                if v > qtr_stockout.mean() else GREEN for v in qtr_stockout.values]
ax.bar(range(len(qtr_stockout)), qtr_stockout.values,
       color=bar_colors_q, edgecolor="white", width=0.7)
ax.axhline(qtr_stockout.mean(), color=DARK, linestyle="--", lw=1.2,
           label=f"Avg {qtr_stockout.mean():.0f}")
ax.set_xticks(range(len(qtr_stockout)))
ax.set_xticklabels(qtr_stockout.index, rotation=40, ha="right", fontsize=8)
ax.set_title("Stockout Events by Quarter")
ax.set_ylabel("Stockout Count")
ax.legend(fontsize=9)

# 4f. Disruption cost by category
ax = axes[1, 2]
cat_cost = df.groupby("category")["disruption_cost_usd"].sum().sort_values(ascending=True)
bars = ax.barh(cat_cost.index, cat_cost.values / 1000,
               color=BLUE, edgecolor="white", height=0.55)
for bar, val in zip(bars, cat_cost.values):
    ax.text(val / 1000 + 0.5, bar.get_y() + bar.get_height()/2,
            f"${val/1000:.0f}K", va="center", fontsize=9)
ax.set_title("Total Disruption Cost by Category")
ax.set_xlabel("Disruption Cost ($K)")

plt.tight_layout()
plt.savefig("/home/claude/supply-chain-risk/eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓  Saved: eda_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHART 2 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Lead Time Anomaly Detection — Supplier Risk Signals", fontsize=15, fontweight="bold", y=1.01)

high_risk_sups = risk_df[risk_df["risk_tier"].isin(["Critical", "High"])]["supplier_id"].tolist()
plot_sups = high_risk_sups[:4]

for ax, sid in zip(axes.flatten(), plot_sups):
    sdf = df[df["supplier_id"] == sid].sort_values("date")
    name = sdf["supplier_name"].iloc[0]
    tier = risk_df[risk_df["supplier_id"] == sid]["risk_tier"].iloc[0]

    ax.plot(sdf["date"], sdf["lead_time_days"], color=GRAY, lw=1, alpha=0.7, label="Lead time")
    ax.plot(sdf["date"], sdf["rolling_lt_4w"],  color=BLUE, lw=2, label="4-week rolling avg")

    mu  = sdf["lt_mean"].iloc[0]
    sig = sdf["lt_std"].iloc[0]
    ax.axhline(mu + 2 * sig, color=RED, linestyle="--", lw=1.2, label="Anomaly threshold (μ+2σ)")
    ax.axhline(mu,           color=GREEN, linestyle=":",  lw=1.2, label="Mean")

    anomalies = sdf[sdf["anomaly"]]
    ax.scatter(anomalies["date"], anomalies["lead_time_days"],
               color=RED, s=40, zorder=5, label=f"{len(anomalies)} anomalies")

    ax.set_title(f"{name}  [{tier}]",
                 color=RISK_COLORS[tier], fontweight="bold")
    ax.set_ylabel("Lead Time (days)")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig("/home/claude/supply-chain-risk/anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓  Saved: anomaly_detection.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CHART 3 — RISK SCORECARD + FINANCIAL IMPACT
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Supplier Risk Scorecard & Financial Impact", fontsize=15, fontweight="bold")

# 6a. Composite risk score bar
ax1 = fig.add_subplot(gs[0, :2])
colors_rs = [RISK_COLORS[t] for t in risk_df["risk_tier"]]
bars = ax1.barh(risk_df["supplier_name"], risk_df["risk_score"],
                color=colors_rs, edgecolor="white", height=0.6)
for bar, row in zip(bars, risk_df.itertuples()):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{row.risk_score:.0f}  [{row.risk_tier}]",
             va="center", fontsize=9,
             color=RISK_COLORS[row.risk_tier], fontweight="bold")
ax1.axvline(40, color=AMBER, linestyle="--", lw=1.2, alpha=0.7, label="High risk threshold (40)")
ax1.axvline(60, color=RED,   linestyle="--", lw=1.2, alpha=0.7, label="Critical threshold (60)")
ax1.set_title("Composite Risk Score by Supplier (0–100)")
ax1.set_xlabel("Risk Score")
ax1.set_xlim(0, 105)
ax1.legend(fontsize=9)
patches = [mpatches.Patch(color=v, label=k) for k, v in RISK_COLORS.items()]
ax1.legend(handles=patches, fontsize=8, loc="lower right")

# 6b. Risk tier donut
ax2 = fig.add_subplot(gs[0, 2])
tier_counts = risk_df["risk_tier"].value_counts()
tier_order  = ["Critical", "High", "Medium", "Low"]
sizes  = [tier_counts.get(t, 0) for t in tier_order]
clrs   = [RISK_COLORS[t] for t in tier_order]
wedges, texts, autotexts = ax2.pie(
    [s for s in sizes if s > 0],
    labels=[t for t, s in zip(tier_order, sizes) if s > 0],
    colors=[c for c, s in zip(clrs, sizes) if s > 0],
    autopct="%1.0f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
    at.set_color("white")
centre = plt.Circle((0, 0), 0.5, fc="white")
ax2.add_patch(centre)
ax2.set_title("Supplier Risk Distribution")

# 6c. Score components stacked bar
ax3 = fig.add_subplot(gs[1, :2])
components = ["delay_score", "defect_score", "stockout_score", "anomaly_score"]
comp_labels = ["Delay (30%)", "Defect (25%)", "Stockout (25%)", "Anomaly (20%)"]
comp_colors = [RED, AMBER, BLUE, GRAY]
bottoms = np.zeros(len(risk_df))
for comp, label, clr in zip(components, comp_labels, comp_colors):
    ax3.barh(risk_df["supplier_name"], risk_df[comp],
             left=bottoms, color=clr, label=label, edgecolor="white", height=0.55)
    bottoms += risk_df[comp].values
ax3.set_title("Risk Score Components Breakdown")
ax3.set_xlabel("Score Contribution")
ax3.legend(fontsize=9, loc="lower right")

# 6d. Disruption cost vs risk score
ax4 = fig.add_subplot(gs[1, 2])
for _, row in risk_df.iterrows():
    ax4.scatter(row["risk_score"], row["total_disruption_cost"] / 1000,
                s=120, color=RISK_COLORS[row["risk_tier"]],
                edgecolors="white", lw=0.8, zorder=3)
    ax4.annotate(row["supplier_name"].split("a")[0] + "...",
                 (row["risk_score"], row["total_disruption_cost"] / 1000),
                 fontsize=7.5, xytext=(4, 3), textcoords="offset points")
ax4.set_title("Risk Score vs Disruption Cost")
ax4.set_xlabel("Risk Score")
ax4.set_ylabel("Total Disruption Cost ($K)")
ax4.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:.0f}K"))

plt.savefig("/home/claude/supply-chain-risk/risk_scorecard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓  Saved: risk_scorecard.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TIME-SERIES TREND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("TIME-SERIES TREND — QUARTERLY SUMMARY")
print("=" * 65)

qtr_summary = df.groupby("quarter").agg(
    total_orders         = ("order_value_usd",      "count"),
    on_time_rate         = ("on_time_delivery",      "mean"),
    avg_delay            = ("delay_days",            "mean"),
    stockouts            = ("stockout_event",        "sum"),
    disruption_cost      = ("disruption_cost_usd",   "sum"),
    anomaly_events       = ("anomaly",               "sum"),
).reset_index()
qtr_summary["on_time_rate"] = qtr_summary["on_time_rate"].map("{:.1%}".format)
qtr_summary["avg_delay"]    = qtr_summary["avg_delay"].map("{:.1f}d".format)
qtr_summary["disruption_cost"] = qtr_summary["disruption_cost"].map("${:,.0f}".format)
print(qtr_summary.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. SQL-STYLE AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SQL-STYLE RISK AGGREGATIONS")
print("=" * 65)

# Q1: Top 5 costliest disruption incidents (weeks)
print("\nTop 5 Costliest Disruption Weeks:")
top_incidents = (
    df[df["disruption_cost_usd"] > 0]
    .sort_values("disruption_cost_usd", ascending=False)
    .head(5)[["date", "supplier_name", "region", "delay_days",
              "stockout_event", "disruption_cost_usd"]]
)
top_incidents["disruption_cost_usd"] = top_incidents["disruption_cost_usd"].map("${:,.0f}".format)
top_incidents["delay_days"]          = top_incidents["delay_days"].map("{:.1f}d".format)
print(top_incidents.to_string(index=False))

# Q2: Region-level risk summary
print("\nRegion-Level Risk Summary:")
region_risk = (
    df.groupby("region").agg(
        suppliers     = ("supplier_id",         "nunique"),
        on_time_rate  = ("on_time_delivery",     "mean"),
        avg_delay     = ("delay_days",           "mean"),
        total_stockouts=("stockout_event",       "sum"),
        total_cost    = ("disruption_cost_usd",  "sum"),
    )
    .sort_values("total_cost", ascending=False)
    .reset_index()
)
region_risk["on_time_rate"] = region_risk["on_time_rate"].map("{:.1%}".format)
region_risk["avg_delay"]    = region_risk["avg_delay"].map("{:.1f}d".format)
region_risk["total_cost"]   = region_risk["total_cost"].map("${:,.0f}".format)
print(region_risk.to_string(index=False))

# Q3: Month-over-month cost change (last 6 months)
print("\nMonth-over-Month Disruption Cost Change (last 6 months):")
mom = (
    df.groupby("year_month")["disruption_cost_usd"].sum()
    .reset_index()
    .tail(7)
)
mom["pct_change"] = mom["disruption_cost_usd"].pct_change() * 100
mom = mom.dropna()
mom["disruption_cost_usd"] = mom["disruption_cost_usd"].map("${:,.0f}".format)
mom["pct_change"]           = mom["pct_change"].map("{:+.1f}%".format)
print(mom.tail(6).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINANCIAL SUMMARY + RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

total_cost   = df["disruption_cost_usd"].sum()
total_stockouts = df["stockout_event"].sum()
worst_sup    = risk_df.iloc[0]["supplier_name"]
worst_region = df.groupby("region")["disruption_cost_usd"].sum().idxmax()

print("\n" + "=" * 65)
print("FINANCIAL IMPACT SUMMARY")
print("=" * 65)
print(f"  Total disruption cost (3 years) : ${total_cost:,.0f}")
print(f"  Annualised disruption cost      : ${total_cost/3:,.0f}")
print(f"  Total stockout events           : {total_stockouts:,}")
print(f"  Highest-risk supplier           : {worst_sup}")
print(f"  Highest-cost region             : {worst_region}")
print()

print("=" * 65)
print("RISK MITIGATION RECOMMENDATIONS")
print("=" * 65)

recommendations = [
    ("SUP-001 & SUP-005 (Critical tier)",
     "Immediately dual-source — identify backup suppliers to reduce single-vendor dependency."),
    ("Seasonal Q3 disruption pattern",
     "Pre-build 6-week safety stock for Asia-region suppliers every May before peak disruption window."),
    ("Electronic check payment → delayed orders",
     "Enforce SLA-backed contracts with financial penalties for delays > 5 days."),
    ("Anomaly detection integration",
     "Deploy μ+2σ threshold monitoring as a weekly automated alert to procurement team."),
    ("Region diversification",
     f"{worst_region} accounts for the highest disruption cost — explore Vietnam/India alternatives to de-risk."),
]

for i, (context, rec) in enumerate(recommendations, 1):
    print(f"\n  REC {i} [{context}]")
    print(f"  → {rec}")

print("\n\n✅  Analysis complete. Charts saved to /home/claude/supply-chain-risk/")
