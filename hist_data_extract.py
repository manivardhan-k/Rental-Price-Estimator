import pandas as pd

# Load CSV
df = pd.read_csv("Metro_zori_uc_sfrcondomfr_sm_month.csv")

# Lowercase state codes
df['state'] = df['StateName'].str.lower()

# Keep only monthly columns from 2020–2025
years = ['2020','2021','2022','2023','2024','2025']
monthly_cols = [c for c in df.columns if any(c.startswith(y) for y in years)]

# Convert to numeric just in case
df[monthly_cols] = df[monthly_cols].apply(pd.to_numeric, errors='coerce')

# Create yearly columns by averaging monthly data
for y in years:
    year_cols = [c for c in monthly_cols if c.startswith(y)]
    df[y] = df[year_cols].mean(axis=1)

# Keep only state + yearly averages
df_yearly = df[['state'] + years]

# Group by state (in case multiple regions per state)
df_yearly = df_yearly.groupby('state').mean().reset_index()

# df_yearly.to_csv("state_historic.csv", index=False)

print(df_yearly.info())
print(df_yearly.head())
