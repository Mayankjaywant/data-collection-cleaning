import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/sample_digital_public_services.csv")
OUT = Path("data/processed/cleaned_digital_public_services.csv")

df = pd.read_csv(RAW)

# Standardize column names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_", regex=False)
)

# Strip text fields
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# Convert date
df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")

# Validate numeric ranges
df.loc[~df["service_awareness"].between(0, 4), "service_awareness"] = np.nan
df.loc[~df["service_usage"].between(0, 4), "service_usage"] = np.nan
df.loc[~df["satisfaction_score"].between(1, 5), "satisfaction_score"] = np.nan
df.loc[~df["completion_success"].isin([0, 1]), "completion_success"] = np.nan

# Remove exact duplicate rows
df = df.drop_duplicates()

# Validate unique IDs
duplicate_ids = df[df["record_id"].duplicated(keep=False)]
if not duplicate_ids.empty:
    print("Review duplicate record_id values:")
    print(duplicate_ids[["record_id"]])

# Quality report
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nMissing values:")
print(df.isna().sum())
print("\nCategory checks:")
print(df["primary_barrier"].value_counts(dropna=False))

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\nSaved cleaned file to: {OUT}")
