import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataweb.csv")

df = pd.read_csv(csv_path, sep=';')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("First rows:")
print(df.head())

# Remove rows with NaN values
df = df.dropna()

# Check duplicate rows
print('\nDuplicated rows:')
print(df.duplicated().sum())

# Filter only products starting with 85 (Electrical/Electronic Products)
df = df[df["HTS Number"].astype(str).str.startswith("85")].copy()
print(f"\nRows after filtering HTS 85: {df.shape[0]}")

# Separate the two data types
imports = df[df["Data Type"] == "Customs Value"].copy()
duties = df[df["Data Type"] == "Calculated Duties"].copy()

imports = imports.drop(columns=["Data Type"])
duties = duties.drop(columns=["Data Type"])

# Wide -> Long
imports_long = imports.melt(
    id_vars=["Country", "HTS Number", "Description"], 
    var_name="year", 
    value_name="imports" 
)

duties_long = duties.melt(
    id_vars=["Country", "HTS Number", "Description"],
    var_name="year",
    value_name="duties"
)

# Merge the two datasets
panel = imports_long.merge(
    duties_long,
    on=["Country", "HTS Number", "year"],
    how="inner"
)

# Clean numbers (since they have American format with commas)
panel["imports"] = pd.to_numeric(panel["imports"].astype(str).str.replace(",", ""), errors='coerce')
panel["duties"] = pd.to_numeric(panel["duties"].astype(str).str.replace(",", ""), errors='coerce')

# Remove rows with zero imports
panel = panel[panel["imports"] > 0] # approximately 40% of rows

# Effective tariff rate  
panel["tariff"] = panel["duties"] / panel["imports"]

# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# 1. DESCRIPTIVE STATISTICS
print("\n1. GENERAL DESCRIPTIVE STATISTICS")
print("-"*60)
print("\nDataset dimensions:", panel.shape)
print(f"Time period: {panel['year'].min()} - {panel['year'].max()}")
print(f"Number of countries: {panel['Country'].nunique()}")
print(f"Number of HTS products: {panel['HTS Number'].nunique()}")

# Statistics for key variables
print("\n2. MAIN VARIABLES STATISTICS")
print("-"*60)
print("\nImports ($):")
print(panel['imports'].describe())
print(f"\nRange: ${panel['imports'].min():,.0f} - ${panel['imports'].max():,.0f}")
print(f"Total imports for period: ${panel['imports'].sum():,.0f}")

print("\nDuties paid ($):")
print(panel['duties'].describe())

print("\nEffective tariff (%):")
print((panel['tariff'] * 100).describe())

# 3. DISTRIBUTION BY COUNTRY AND YEAR
print("\n3. DATA DISTRIBUTION BY COUNTRY")
print("-"*60)
country_stats = panel.groupby('Country').agg({
    'imports': ['count', 'sum', 'mean'],
    'duties': 'sum',
    'tariff': 'mean'
}).round(2)
print(country_stats)

# 4. PLOTS
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Import distribution (log scale)
ax1 = axes[0, 0]
panel['log_imports'] = np.log10(panel['imports'] + 1)
ax1.hist(panel['log_imports'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Log10(Imports + 1)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Distribution of Imports (Log Scale)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Duties distribution (log scale)
ax2 = axes[0, 1]
panel['log_duties'] = np.log10(panel['duties'] + 1)
ax2.hist(panel['log_duties'], bins=50, color='orange', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Log10(Duties + 1)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Distribution of Duties (Log Scale)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Effective tariff distribution
ax3 = axes[0, 2]
ax3.hist(panel['tariff'] * 100, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Effective Tariff (%)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Distribution of Effective Tariff (%)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Top 10 products by import value
ax4 = axes[1, 0]
top_products = panel.groupby('HTS Number')['imports'].sum().nlargest(10)
ax4.barh(range(len(top_products)), top_products.values/1e9, color='green', alpha=0.7)
ax4.set_yticks(range(len(top_products)))
ax4.set_yticklabels([str(x) for x in top_products.index], fontsize=8)
ax4.set_xlabel('Total Imports (Billions $)', fontsize=10)
ax4.set_title('Top 10 Products by Import Value', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Top 10 products by duties paid
ax5 = axes[1, 1]
top_duties = panel.groupby('HTS Number')['duties'].sum().nlargest(10)
ax5.barh(range(len(top_duties)), top_duties.values/1e9, color='darkred', alpha=0.7)
ax5.set_yticks(range(len(top_duties)))
ax5.set_yticklabels([str(x) for x in top_duties.index], fontsize=8)
ax5.set_xlabel('Total Duties (Billions $)', fontsize=10)
ax5.set_title('Top 10 Products by Duties Paid', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Box plot effective tariff by country
ax6 = axes[1, 2]
tariff_data = [panel[panel['Country'] == c]['tariff']*100 for c in panel['Country'].unique()]
bp = ax6.boxplot(tariff_data, tick_labels=panel['Country'].unique(), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax6.set_ylabel('Effective Tariff Rate (%)', fontsize=10)
ax6.set_title('Effective Tariff Rate by Country', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================
# Q1 - EVOLUTION OF TARIFFS (China vs Vietnam)
# ============================================================
panel_cv = panel.copy()
panel_cv["year"] = panel_cv["year"].astype(int)

# Calculate weighted average tariff by imports
panel_cv["weighted_tariff"] = panel_cv["tariff"] * panel_cv["imports"]
tariff_evolution = panel_cv.groupby(["Country", "year"]).agg({
    "weighted_tariff": "sum",
    "imports": "sum"
}).reset_index()
tariff_evolution["avg_tariff"] = (tariff_evolution["weighted_tariff"] / tariff_evolution["imports"]) * 100

# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=tariff_evolution, x="year", y="avg_tariff", hue="Country", marker="o", linewidth=2)
plt.axvline(x=2018, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Trade War (2018)")
plt.title("Evolution of Effective Tariffs", fontsize=14, fontweight="bold")
plt.xlabel("Year", fontsize=11)
plt.ylabel("Average Effective Tariff (%)", fontsize=11)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Descriptive statistics pre/post 2018
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

for country in tariff_evolution["Country"].unique():
    data_country = tariff_evolution[tariff_evolution["Country"] == country]
    pre_2018 = data_country[data_country["year"] < 2018]["avg_tariff"]
    post_2018 = data_country[data_country["year"] >= 2018]["avg_tariff"]
    
    print(f"\n{country}:")
    print(f"  Pre-2018  (2002-2017): Mean={pre_2018.mean():.2f}%, Min={pre_2018.min():.2f}%, Max={pre_2018.max():.2f}%")
    print(f"  Post-2018 (2018-2024): Mean={post_2018.mean():.2f}%, Min={post_2018.min():.2f}%, Max={post_2018.max():.2f}%")
    print(f"  Change: {post_2018.mean() - pre_2018.mean():+.2f} percentage points")

# ============================================================
# Q2 - HAVE TARIFFS REDUCED IMPORTS FROM CHINA?
# ============================================================
print("\n" + "="*60)
print("Q2 - REGRESSION ANALYSIS: IMPACT OF TARIFFS ON IMPORTS")
print("="*60)

analysis_data = panel_cv.groupby(["Country", "year"]).agg({
    "imports": "sum",
    "weighted_tariff": "sum"
}).reset_index()
analysis_data["tariff_pct"] = (analysis_data["weighted_tariff"] / analysis_data["imports"]) * 100
analysis_data["log_imports"] = np.log(analysis_data["imports"])
analysis_data["post_2018"] = (analysis_data["year"] >= 2018).astype(int)
analysis_data["china"] = (analysis_data["Country"] == "China").astype(int)

# Correlation analysis
china_data = analysis_data[analysis_data["Country"] == "China"]
corr, p_value = pearsonr(china_data["tariff_pct"], china_data["log_imports"])
print(f"\nCorrelation (China): {corr:.4f} (p-value: {p_value:.4f})")

# MODEL 1: Basic regression - log(Imports) = β₀ + β₁*tariff_pct + ε
print("\n" + "-"*60)
print("MODEL 1: Basic Regression (China only) - log(Imports) = β₀ + β₁*tariff_pct + ε")
print("-"*60)
X1 = china_data[["tariff_pct"]]
X1 = sm.add_constant(X1)
y1 = china_data["log_imports"]
model1 = sm.OLS(y1, X1).fit()
print(model1.summary())

# MODEL 2: Adding post_2018 dummy (China only) - log(Imports) = β₀ + β₁*tariff_pct + β₂*post_2018 + ε
print("\n" + "-"*60)
print("MODEL 2: Adding post_2018 dummy (China only) - log(Imports) = β₀ + β₁*tariff_pct + β₂*post_2018 + ε")
print("-"*60)
X2 = china_data[["tariff_pct", "post_2018"]]
X2 = sm.add_constant(X2)
model2 = sm.OLS(y1, X2).fit()
print(model2.summary())

# MODEL 3: Complete model (China and Vietnam with interactions) - log(Imports) = β₀ + β₁*tariff_pct + β₂*post_2018 + β₃*china + ε
print("\n" + "-"*60)
print("MODEL 3: Complete model (China and Vietnam with interactions) - log(Imports) = β₀ + β₁*tariff_pct + β₂*post_2018 + β₃*china + ε")
print("-"*60)
X3 = analysis_data[["tariff_pct", "post_2018", "china"]]
X3 = sm.add_constant(X3)
y3 = analysis_data["log_imports"]
model3 = sm.OLS(y3, X3).fit()
print(model3.summary())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: China: Tariffs vs Log(Imports)
ax1 = axes[0]
pre = china_data[china_data["post_2018"] == 0]
post = china_data[china_data["post_2018"] == 1]
ax1.scatter(pre["tariff_pct"], pre["log_imports"], label="Pre-2018", alpha=0.7, s=100, color="blue")
ax1.scatter(post["tariff_pct"], post["log_imports"], label="Post-2018", alpha=0.7, s=100, color="red")
ax1.plot(china_data["tariff_pct"], model1.predict(X1), color="black", linestyle="--", linewidth=2, label="Regression Line")
ax1.set_xlabel("Effective Tariff (%)", fontsize=11)
ax1.set_ylabel("Log(Imports)", fontsize=11)
ax1.set_title("China: Tariffs vs Log(Imports)", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: China vs Vietnam
ax2 = axes[1]
for country, color in [("China", "red"), ("Vietnam", "green")]:
    data = analysis_data[analysis_data["Country"] == country]
    ax2.scatter(data["tariff_pct"], data["log_imports"], label=country, alpha=0.7, s=100, color=color)
ax2.set_xlabel("Effective Tariff (%)", fontsize=11)
ax2.set_ylabel("Log(Imports)", fontsize=11)
ax2.set_title("China vs Vietnam: Tariffs vs Imports", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Q3 - TRADE DIVERSION TOWARD VIETNAM
# ============================================================
print("\n" + "="*60)
print("Q3 – TRADE DIVERSION: REALLOCATION TOWARD VIETNAM (LOG SCALE)")
print("="*60)

# Calculate total imports
imports_mi = (
    analysis_data
    .groupby(["year", "Country"])["imports"]
    .sum()
    .sort_index()
)

# Pivot and logarithmic transformation
imports_wide = imports_mi.unstack("Country")

# Apply natural logarithm
imports_wide["log_China"] = np.log(imports_wide["China"])
imports_wide["log_Vietnam"] = np.log(imports_wide["Vietnam"])

# Calculate market shares (remain in % for interpretive clarity)
imports_wide["total"] = imports_wide[["China", "Vietnam"]].sum(axis=1)
imports_wide["china_share"] = (imports_wide["China"] / imports_wide["total"]) * 100

# Statistics pre/post 2018 on LOG
pre = imports_wide.loc[imports_wide.index < 2018]
post = imports_wide.loc[imports_wide.index >= 2018]

log_china_pre = pre["log_China"].mean()
log_china_post = post["log_China"].mean()
log_vietnam_pre = pre["log_Vietnam"].mean()
log_vietnam_post = post["log_Vietnam"].mean()

print(f"\nLOG AVERAGE IMPORTS (Natural Log):")
print(f"  China:")
print(f"    Pre-2018:  {log_china_pre:.4f}")
print(f"    Post-2018: {log_china_post:.4f} (Diff: {log_china_post - log_china_pre:.4f})")
print(f"  Vietnam:")
print(f"    Pre-2018:  {log_vietnam_pre:.4f}")
print(f"    Post-2018: {log_vietnam_post:.4f} (Diff: {log_vietnam_post - log_vietnam_pre:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Log-Imports Trend
ax = axes[0]
imports_wide[["log_China", "log_Vietnam"]].plot(ax=ax, marker='o', linewidth=2)
ax.axvline(2018, color='red', linestyle='--', alpha=0.7, label='Trade War (2018)')
ax.set_title("Log of US imports: China vs Vietnam", fontsize=12, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Log scale (ln$)", fontsize=11)
ax.legend(["ln(China)", "ln(Vietnam)", "2018 Threshold"])
ax.grid(True, linestyle=':', alpha=0.6)

# Plot 2: China's market share
ax = axes[1]
imports_wide["china_share"].plot(ax=ax, marker='o', color='blue', linewidth=2)
ax.axvline(2018, color='red', linestyle='--', alpha=0.7)
ax.axhline(pre["china_share"].mean(), linestyle=':', color='gray', label='Mean Pre-2018')
ax.axhline(post["china_share"].mean(), linestyle=':', color='black', label='Mean Post-2018')
ax.set_title("China's market share (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

fig.suptitle("Trade diversion analysis", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# Statistical test on LOG (verify if the log mean has changed significantly)
t_stat, p_value = stats.ttest_ind(
    pre["log_China"], 
    post["log_China"], 
    equal_var=False
)

print("\nSTATISTICAL TEST ON LOG(CHINA):")
print(f"T-statistic = {t_stat:.4f}")
print(f"P-value     = {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant change in log-imports.")
else:
    print("Result: No statistically significant change.")

# Statistical test on China's Market Share
t_stat_share, p_val_share = stats.ttest_ind(
    pre["china_share"], 
    post["china_share"], 
    equal_var=False
)

print(f"\nSTATISTICAL TEST ON CHINA MARKET SHARE:")
print(f"T-statistic = {t_stat_share:.4f}")
print(f"P-value     = {p_val_share:.4f}")

# ============================================================
# Q4 - DIFFERENCE-IN-DIFFERENCES: TRADE DIVERSION CAUSALITY
# ============================================================
print("\n" + "="*60)
print("Q4 - DIFFERENCE-IN-DIFFERENCES (DiD)")
print("="*60)

# Create interaction term
analysis_data["china_post"] = analysis_data["china"] * analysis_data["post_2018"]

# DiD model with log_imports as dependent variable
# We use the previously calculated log_imports
y = analysis_data["log_imports"]
X_did = analysis_data[["china", "post_2018", "china_post"]]
X_did = sm.add_constant(X_did)

# Fit the model
model_did = sm.OLS(y, X_did).fit()
print("\nMODEL DiD:")
print(model_did.summary())

# Interpretation
beta_did = model_did.params["china_post"]
p_did = model_did.pvalues["china_post"]

print(f"\n{'='*60}")
print(f"CAUSAL EFFECT OF TRADE WAR (DiD)")
print(f"{'='*60}")
print(f"β₃ (china × post_2018) = {beta_did:.4f}")
print(f"p-value = {p_did:.4f}")

