import pandas as pd

# https://www.zillow.com/research/data/

# Load CSV
df = pd.read_csv("csvs/Metro_zori_uc_sfrcondomfr_sm_month.csv")

# Lowercase state codes
df['state'] = df['StateName'].str.lower().str.strip().fillna('US')

# Select monthly columns 2015–2025
years = ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025']
monthly_cols = [c for c in df.columns if any(c.startswith(y) for y in years)]

# Convert to numeric
df[monthly_cols] = df[monthly_cols].apply(pd.to_numeric, errors='coerce')

# Build yearly averages in a separate frame
yearly_avg = pd.concat(
    {y: df[[c for c in monthly_cols if c.startswith(y)]].mean(axis=1)
     for y in years},
    axis=1
)

# Combine state + yearly averages
df_yearly = pd.concat([df[['state']], yearly_avg], axis=1)

# Group by state
df_yearly = df_yearly.groupby('state', as_index=False).mean()

# df_yearly.to_csv("csvs/state_historic.csv", index=False)
# Did not update the csv with US historic data - not necessary

print(df_yearly.info())
print(df_yearly.head())
