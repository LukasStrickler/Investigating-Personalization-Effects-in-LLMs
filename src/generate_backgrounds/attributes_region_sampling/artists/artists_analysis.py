import json
from pathlib import Path

import pandas as pd

folder = Path(__file__).parent

# Read with explicit UTF-8 encoding ("utf-8-sig" also handles Excel's BOM)
df = pd.read_csv(folder / "artists_list_shuffled.csv", encoding="utf-8-sig")

# Keep only selected artists (handles TRUE as text or boolean)
selected = df[df["Selected"].astype(str).str.upper() == "TRUE"]

# Keep only region and artist columns
result = (
    selected[["Region", "Artist Name"]]
    .rename(columns={"Region": "region", "Artist Name": "artist"})
    .reset_index(drop=True)
)

# Save CSV with utf-8-sig so Excel displays special characters correctly
result.to_csv(folder / "selected_artists.csv", index=False, encoding="utf-8-sig")

# Save JSON grouped by region, keeping real characters (no \uXXXX escapes)
by_region = result.groupby("region")["artist"].apply(list).to_dict()
with open(folder / "selected_artists.json", "w", encoding="utf-8") as f:
    json.dump(by_region, f, ensure_ascii=False, indent=2)
