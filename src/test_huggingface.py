from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Create a tiny test dataset
# ---------------------------------------------------------

data = {
    "city": [
        "Colombo",
        "London",
        "Tokyo",
        "Cairo",
        "Sydney",
    ],
    "temperature": [
        28.4,
        11.2,
        17.8,
        24.3,
        19.1,
    ],
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Save as Parquet
# ---------------------------------------------------------

output_dir = Path("data/test")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "hf_test.parquet"

df.to_parquet(output_file, index=False)

print(f"Created: {output_file}")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
