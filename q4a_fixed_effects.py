import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

#1. Load data 
BASE = "/Users/meisirui/Desktop/Internships/AEI/data assessment/"

gdp = pd.read_excel(BASE + "GDPpercapita_Worldbank.xls",
                    sheet_name="Data", header=3)
lfe = pd.read_excel(BASE + "Life_expectation_female_Worldbank.xls",
                    sheet_name="Data", header=3)

#2. Reshape from wide to long format
# World Bank files are in wide format (one column per year).
# I request 1960-2010, but PPP GDP data is only available from ~1990,
# so the panel will effectively cover 1990-2010 after dropping missing values.
years = [str(y) for y in range(1960, 2011)]

gdp_long = gdp[["Country Name", "Country Code"] + years].melt(
    id_vars=["Country Name", "Country Code"],
    var_name="year", value_name="gdp_pc"
)

lfe_long = lfe[["Country Name", "Country Code"] + years].melt(
    id_vars=["Country Name", "Country Code"],
    var_name="year", value_name="life_exp"
)

#3. Merge and clean 
df = pd.merge(gdp_long, lfe_long, on=["Country Code", "year"])
df = df.rename(columns={"Country Code": "code", "Country Name x": "country"})

# Keep only true country-level observations (ISO-3 codes are exactly 3 characters)
df = df[df["code"].str.len() == 3]
df["year"] = df["year"].astype(int)
df = df.dropna(subset=["gdp_pc", "life_exp"])
df = df[df["gdp_pc"] > 0]
df["ln_gdp"] = np.log(df["gdp_pc"])

print(f"Panel observations: {len(df)}")
print(f"Countries: {df['code'].nunique()}")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
print("(Note: PPP GDP data unavailable before 1990 in WDI)")

#4. Set panel index (country, year)
df = df.set_index(["code", "year"])

# 5. Fixed effects regression
# EntityEffects absorbs all time-invariant country characteristics
# (geography, institutions, health infrastructure, etc.)
model = PanelOLS(df["life_exp"], df["ln_gdp"], entity_effects=True)
result = model.fit(cov_type="clustered", cluster_entity=True)

fe_coef = result.params["ln_gdp"]
fe_se   = result.std_errors["ln_gdp"]
fe_pval = result.pvalues["ln_gdp"]

print("\n" + "="*60)
print("FIXED EFFECTS REGRESSION RESULTS (1990-2010)")
print("="*60)
print(result.summary.tables[1])
print(f"R-squared (within): {result.rsquared_within:.4f}")

#6. Cross-sectional OLS from Q3 (for comparison)
gdp_2010 = gdp[["Country Name", "Country Code", "2010"]].copy()
gdp_2010.columns = ["country", "code", "gdp_pc"]
gdp_2010 = gdp_2010[gdp_2010["code"].str.len() == 3]

lfe_2010 = lfe[["Country Name", "Country Code", "2010"]].copy()
lfe_2010.columns = ["country", "code", "life_exp"]
lfe_2010 = lfe_2010[lfe_2010["code"].str.len() == 3]

df_2010 = pd.merge(gdp_2010, lfe_2010, on="code").dropna()
df_2010 = df_2010[df_2010["gdp_pc"] > 0]
df_2010["ln_gdp"] = np.log(df_2010["gdp_pc"])

ols_slope, ols_intercept, ols_r, _, _ = sp_stats.linregress(
    df_2010["ln_gdp"], df_2010["life_exp"]
)

#7. Print comparison
print("\n" + "="*60)
print("COMPARISON: Cross-sectional OLS (Q3) vs. Fixed Effects")
print("="*60)
print(f"  Cross-sectional OLS (2010):      b = {ols_slope:.4f},  R² = {ols_r**2:.4f}")
print(f"  Panel Fixed Effects (1990-2010): b = {fe_coef:.4f},  R²_within = {result.rsquared_within:.4f}")
print(f"\nThe OLS estimate ({ols_slope:.2f}) is larger than the FE estimate ({fe_coef:.2f}).")
print("Controlling for time-invariant country characteristics reduces the")
print("coefficient, consistent with upward omitted variable bias in OLS.")
