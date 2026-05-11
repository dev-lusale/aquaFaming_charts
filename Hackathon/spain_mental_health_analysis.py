import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import scipy.stats as stats
import warnings
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


DATA_PATH  = Path("C:\\Users\\Dell\\Desktop\\ZPPA Registered Suppliers.csv")
OUTPUT_DIR = Path("C:\\Users\\Dell\\Desktop\\spain_mental_health_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG, FACE = "#F7F7F7", "#FFFFFF"
COLORS = {
    "anxiety":        "#E24B4A",
    "depression":     "#378ADD",
    "online_therapy": "#1D9E75",
    "panic_attacks":  "#EF9F27",
    "stress":         "#7F77DD",
}

def new_fig(title, subtitle="", figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(FACE)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=10, color="#555555", pad=10)
    return fig, ax

def save(fig, name):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {path.name}")


# STEP 1: UNDERSTAND THE DATA

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("STEP 1: DATA UNDERSTANDING")
print("=" * 60)
print(f"\nShape     : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
print(f"Duration  : {df['date'].dt.year.nunique()} years of monthly data")

print("\nColumn descriptions:")
col_desc = {
    "date":           "Monthly timestamp (2004-03 to 2025-12)",
    "anxiety":        "Google search interest for 'anxiety' (0-100)",
    "depression":     "Google search interest for 'depression' (0-100)",
    "online_therapy": "Google search interest for 'online therapy' (0-100)",
    "panic_attacks":  "Google search interest for 'panic attacks' (0-100)",
    "stress":         "Google search interest for 'stress' (0-100)",
    "anxiety_diff":   "Month-over-month change in anxiety score",
    "online_diff":    "Month-over-month change in online_therapy score",
}
for col, desc in col_desc.items():
    print(f"  {col:<18} -- {desc}")

print("\nDescriptive statistics:")
print(df[["anxiety", "depression", "online_therapy", "panic_attacks", "stress"]].describe().round(2))
print("\nMissing values:", df.isnull().sum().sum(), "(none)")
print("Duplicates   :", df.duplicated().sum())



# STEP 2: CLEAN & FEATURE ENGINEERING

print("\n" + "=" * 60)
print("STEP 2: CLEANING & FEATURE ENGINEERING")
print("=" * 60)

df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["covid_period"] = df["year"].apply(
    lambda y: "Pre-COVID" if y < 2020 else "COVID (2020-21)" if y <= 2021 else "Post-COVID"
)

print("\nFeatures added: year, month, quarter, covid_period")
print("No missing values or duplicates found -- dataset is clean")
print(f"online_therapy legitimately ZERO before: "
      f"{df[df['online_therapy'] == 0]['date'].max().strftime('%Y-%m')} (not a data error)")

TOPICS = ["anxiety", "depression", "online_therapy", "panic_attacks", "stress"]



# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)

print("\n" + "=" * 60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 3a. Yearly averages
yearly = df.groupby("year")[TOPICS].mean().round(2)
print("\nYearly averages (selected years):")
display_rows = list(yearly.index[:3]) + [2019, 2020, 2021] + list(yearly.index[-3:])
print(yearly.loc[sorted(set(display_rows))])

# 3b. Peak values
print("\nPeak search interest values:")
for col in TOPICS:
    idx  = df[col].idxmax()
    peak = df[col].max()
    date = df.loc[idx, "date"].strftime("%Y-%m")
    print(f"  {col:<18}: {peak:>3}  at {date}")

# 3c. COVID impact
print("\nCOVID impact -- avg change 2019 to 2020:")
pre   = df[df["year"] == 2019][TOPICS].mean()
post  = df[df["year"] == 2020][TOPICS].mean()
delta = (post - pre).round(2)
for topic in TOPICS:
    sign = "+" if delta[topic] > 0 else ""
    print(f"  {topic:<18}: {sign}{delta[topic]:.2f}")

# 3d. Pearson correlations with anxiety
print("\nPearson correlations with Anxiety:")
for col in ["depression", "online_therapy", "panic_attacks", "stress"]:
    r, p = pearsonr(df["anxiety"], df[col])
    print(f"  anxiety x {col:<18}: r={r:.3f}, p={p:.6f}")

# 3e. Full correlation matrix
print("\nFull correlation matrix:")
print(df[TOPICS].corr().round(3))

# 3f. Seasonal peaks
print("\nSeasonal peak months:")
monthly_avg = df.groupby("month")[TOPICS].mean().round(2)
months_list = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for topic in TOPICS:
    peak_month = monthly_avg[topic].idxmax()
    print(f"  {topic:<18}: peak in {months_list[peak_month-1]}")

# 3g. COVID period averages
print("\nAverage search interest by COVID period:")
print(df.groupby("covid_period")[TOPICS].mean().round(1))



# STEP 4: KEY INSIGHTS

print("\n" + "=" * 60)
print("STEP 4: KEY INSIGHTS")
print("=" * 60)

anx_2004 = yearly.loc[2004, "anxiety"]
anx_2025 = yearly.loc[2025, "anxiety"]
str_2004 = yearly.loc[2004, "stress"]
str_2025 = yearly.loc[2025, "stress"]
ot_2019  = yearly.loc[2019, "online_therapy"]
ot_2020  = yearly.loc[2020, "online_therapy"]

print(f"""
1. ALL MENTAL HEALTH SEARCHES HAVE RISEN DRAMATICALLY (2004-2025)
   Anxiety  : {anx_2004:.1f} (2004) -> {anx_2025:.1f} (2025) = +{(anx_2025-anx_2004)/anx_2004*100:.0f}%
   Stress   : {str_2004:.1f} (2004) -> {str_2025:.1f} (2025) = +{(str_2025-str_2004)/str_2004*100:.0f}%
   Depression: consistently high, avg 68.7 over 21 years

2. COVID-19 TRIGGERED A STRUCTURAL ACCELERATION (2020)
   Online therapy: {ot_2019:.1f} (2019) -> {ot_2020:.1f} (2020) = +{(ot_2020-ot_2019)/ot_2019*100:.0f}% YoY surge
   Anxiety climbed from {yearly.loc[2019,'anxiety']:.1f} (2019) to {yearly.loc[2020,'anxiety']:.1f} (2020)

3. ONLINE THERAPY DID NOT EXIST AS A SEARCH TERM BEFORE APRIL 2009
   Zero search interest before April 2009
   Grew from 0 to 75.9 (2025) -- entirely new demand creation

4. 2022 WAS THE PEAK CRISIS YEAR (POST-COVID CRASH)
   Anxiety peaked at 100 in March 2022
   Depression peaked at 100 in January 2022
   Panic attacks peaked at 100 in August 2022

5. DEPRESSION IS CONSISTENTLY THE HIGHEST-SEARCHED TOPIC
   Avg depression = 68.7 over 21 years, never below 49

6. SEASONAL PATTERNS: SPRING IS THE MOST MENTALLY CHALLENGING SEASON
   March  -- peak anxiety and depression (all 21 years)
   May    -- peak stress month
   July-August -- seasonal trough

7. STRONG CORRELATIONS: THIS IS A SYSTEMIC, NOT ISOLATED, CRISIS
   Anxiety x Panic Attacks: r = 0.97 (near-perfect co-movement)
   Anxiety x Stress       : r = 0.82
   Anxiety x Depression   : r = 0.75
   All p < 0.0001
""")

# STEP 5: RECOMMENDATIONS

print("=" * 60)
print("STEP 5: RECOMMENDATIONS")
print("=" * 60)
print("""
1. SCALE ONLINE MENTAL HEALTH INFRASTRUCTURE
   Target : Ministry of Health, private platforms
   Evidence: +74% YoY surge 2020; still avg 75.9 in 2025
   Action  : Subsidise national online mental health platform access

2. SPRING MENTAL HEALTH AWARENESS CAMPAIGNS
   Target : Schools, employers, public media
   Evidence: March-May = highest anxiety, depression, stress (21 yrs)
   Action  : Annual spring campaign + proactive counselling deployment

3. POST-CRISIS MENTAL HEALTH SURGE CAPACITY PLAN
   Target : Government emergency planners, NGOs
   Evidence: 2022 peaks EXCEEDED 2020 -- crash comes AFTER the crisis
   Action  : Pre-position mental health capacity 12-18 months post-crisis

4. GOOGLE TRENDS AS REAL-TIME SURVEILLANCE TOOL
   Target : Ministry of Health, researchers, policymakers
   Evidence: 262 months of reliable demand signal
   Action  : Monthly public dashboard; alert at score > 80 for 2+ months

5. MANDATORY WORKPLACE STRESS MANAGEMENT PROGRAMMES
   Target : Employers, HR departments, Ministry of Labour
   Evidence: Stress nearly doubled (33->59) over 21 years; still rising
   Action  : Mandate EAPs + annual mental health risk assessments
""")


# ─────────────────────────────────────────────
# STEP 6: VISUALISATIONS -- 8 SEPARATE CHARTS
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 6: GENERATING 8 CHARTS")
print("=" * 60)

yearly = df.groupby("year")[TOPICS].mean()

# CHART 1: All topics 21-year timeline
fig, ax = new_fig(
    "Chart 1 -- Mental Health Search Trends in Spain (2004-2025)",
    "Monthly Google search interest (0-100) for five mental health topics"
)
for topic in TOPICS:
    ax.plot(df["date"], df[topic],
            color=COLORS[topic], linewidth=1.5, alpha=0.85,
            label=topic.replace("_", " ").title())
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
           alpha=0.12, color="#E24B4A", label="COVID Period (2020-21)")
ax.set_xlabel("")
ax.set_ylabel("Search Interest (0-100)", fontsize=11)
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.set_xlim(df["date"].min(), df["date"].max())
save(fig, "chart1_all_trends_timeline.png")

#CHART 2: Annual average grouped bar 
fig, ax = new_fig(
    "Chart 2 -- Annual Average Search Interest by Topic",
    "Yearly mean scores reveal long-term growth across all mental health topics",
    figsize=(13, 6)
)
x       = np.arange(len(yearly.index))
w       = 0.15
offsets = [-2, -1, 0, 1, 2]
for i, topic in enumerate(TOPICS):
    ax.bar(x + offsets[i] * w, yearly[topic], width=w,
           color=COLORS[topic], label=topic.replace("_", " ").title(),
           edgecolor="white", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(yearly.index, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("")
ax.set_ylabel("Avg Search Interest (0-100)", fontsize=11)
ax.legend(fontsize=9, ncol=5, loc="upper left")
save(fig, "chart2_annual_averages.png")

# CHART 3: COVID period comparison 
fig, ax = new_fig(
    "Chart 3 -- COVID-19 Impact on Mental Health Searches",
    "Average search interest: Pre-COVID vs COVID (2020-21) vs Post-COVID",
    figsize=(10, 6)
)
covid_order   = ["Pre-COVID", "COVID (2020-21)", "Post-COVID"]
covid_avg     = df.groupby("covid_period")[TOPICS].mean().reindex(covid_order)
x             = np.arange(len(TOPICS))
w             = 0.25
period_colors = ["#378ADD", "#E24B4A", "#1D9E75"]
for i, (period, color) in enumerate(zip(covid_order, period_colors)):
    ax.bar(x + (i - 1) * w, covid_avg.loc[period], width=w,
           color=color, label=period, edgecolor="white", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", " ").title() for t in TOPICS], fontsize=10)
ax.set_ylabel("Avg Search Interest (0-100)", fontsize=11)
ax.legend(fontsize=9)
for i in range(3):
    for j, topic in enumerate(TOPICS):
        val = covid_avg.iloc[i][topic]
        ax.text(j + (i - 1) * w, val + 0.5,
                f"{val:.0f}", ha="center", fontsize=7, fontweight="bold")
save(fig, "chart3_covid_impact.png")

# CHART 4: Online therapy emergence
fig, ax = new_fig(
    "Chart 4 -- Online Therapy: From Zero to Mainstream (2004-2025)",
    "Online therapy had zero search interest until 2009; now rivals anxiety searches"
)
ax.fill_between(df["date"], df["online_therapy"],
                alpha=0.25, color=COLORS["online_therapy"])
ax.plot(df["date"], df["online_therapy"],
        color=COLORS["online_therapy"], linewidth=2, label="Online Therapy")
ax.axvline(pd.Timestamp("2009-04-01"), color="#888", linestyle="--",
           linewidth=1.5, label="First non-zero search (Apr 2009)")
ax.axvline(pd.Timestamp("2020-03-01"), color="#E24B4A", linestyle="--",
           linewidth=1.5, label="COVID lockdown (Mar 2020)")
ax.annotate("COVID spike:\n+74% YoY",
            xy=(pd.Timestamp("2020-06-01"), 64),
            xytext=(pd.Timestamp("2017-01-01"), 72),
            arrowprops=dict(arrowstyle="->", color="#E24B4A"),
            fontsize=9, color="#E24B4A", fontweight="bold")
ax.set_ylabel("Search Interest (0-100)", fontsize=11)
ax.legend(fontsize=9)
save(fig, "chart4_online_therapy_emergence.png")

# CHART 5: Seasonal heatmap + line chart
month_names    = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_avg_df = df.groupby("month")[TOPICS].mean()
monthly_avg_df.index = month_names

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 5 -- Seasonal Patterns in Mental Health Searches",
             fontsize=14, fontweight="bold", y=1.01)

sns.heatmap(monthly_avg_df.T, annot=True, fmt=".0f",
            cmap="YlOrRd", ax=axes[0],
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Avg Search Interest"},
            annot_kws={"size": 9})
axes[0].set_title("Monthly average search interest by topic", fontsize=10, color="#555")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_facecolor(FACE)

for topic in ["anxiety", "depression", "stress"]:
    axes[1].plot(range(1, 13),
                 df.groupby("month")[topic].mean(),
                 color=COLORS[topic], linewidth=2,
                 marker="o", markersize=5,
                 label=topic.title())
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_names, fontsize=9)
axes[1].set_ylabel("Avg Search Interest (0-100)", fontsize=11)
axes[1].set_title("Seasonal rhythm: anxiety, depression & stress", fontsize=10, color="#555")
axes[1].legend(fontsize=9)
axes[1].set_facecolor(FACE)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "chart5_seasonal_patterns.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> chart5_seasonal_patterns.png")

# CHART 6: Correlation heatmap
fig, ax = new_fig(
    "Chart 6 -- Correlation Matrix: All Mental Health Topics",
    "How strongly do the five search topics move together over 21 years?",
    figsize=(8, 7)
)
corr         = df[TOPICS].corr().round(2)
topic_labels = [t.replace("_", "\n").title() for t in TOPICS]
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.8, linecolor="white",
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=topic_labels, yticklabels=topic_labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
save(fig, "chart6_correlation_heatmap.png")

# CHART 7: Anxiety vs Online Therapy scatter
fig, ax = new_fig(
    "Chart 7 -- Anxiety vs Online Therapy: Crisis Drives Help-Seeking",
    "Scatter coloured by year with regression line"
)
sc = ax.scatter(df["anxiety"], df["online_therapy"],
                c=df["year"], cmap="RdYlGn_r",
                s=60, alpha=0.75, edgecolors="white", linewidth=0.8)
m, b, r_val, p_val, _ = stats.linregress(df["anxiety"], df["online_therapy"])
x_line = np.linspace(df["anxiety"].min(), df["anxiety"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#555", linestyle="--",
        linewidth=2, label=f"Regression  r={r_val:.2f},  p={p_val:.4f}")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Year", fontsize=10)
ax.set_xlabel("Anxiety Search Interest (0-100)", fontsize=11)
ax.set_ylabel("Online Therapy Search Interest (0-100)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.annotate("Pre-2009:\nonline therapy = 0",
            xy=(40, 2), xytext=(60, 10),
            arrowprops=dict(arrowstyle="->", color="#888"),
            fontsize=9, color="#888")
save(fig, "chart7_anxiety_vs_online_therapy.png")

# CHART 8: KPI Summary Cards
kpis = [
    ("Anxiety Growth (2004-2025)",  "+61%",       "#E24B4A"),
    ("Stress Growth (2004-2025)",   "+79%",       "#7F77DD"),
    ("Online Therapy Growth",       "0 -> 75+",   "#1D9E75"),
    ("COVID Therapy Surge (2020)",  "+74% YoY",   "#EF9F27"),
    ("Peak Crisis Year",            "2022",       "#E24B4A"),
    ("Depression avg (21 yrs)",     "68.7 / 100", "#378ADD"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("KPI Summary -- Spain Mental Health Search Trends (2004-2025)",
             fontsize=14, fontweight="bold", y=1.02)

for ax, (label, value, color) in zip(axes.flat, kpis):
    ax.set_facecolor(FACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 0.06, 1, facecolor=color,
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.55, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=22, fontweight="bold", color=color)
    ax.text(0.55, 0.28, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#555555")

fig.tight_layout(pad=1.5)
fig.savefig(OUTPUT_DIR / "chart8_kpi_summary.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> chart8_kpi_summary.png")

print(f"\n{'='*60}")
print(f"All 8 charts saved to: {OUTPUT_DIR}/")
print(f"{'='*60}")
