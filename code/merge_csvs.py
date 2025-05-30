import os
import pandas as pd
import re
from collections import defaultdict

def merge_related_csvs(directory=".", output_directory="merged"):
    if output_directory is None:
        output_directory = directory

    # Group files by common prefix
    groups = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            match = re.match(r"(.+)_\d+\.csv", filename)
            if match:
                prefix = match.group(1)
                groups[prefix].append(os.path.join(directory, filename))

    # Merge files in each group
    for prefix, filepaths in groups.items():
        dfs = [pd.read_csv(f) for f in sorted(filepaths)]
        merged_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(output_directory, f"{prefix}_merged.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Merged {len(filepaths)} files into {output_path}")

# Usage:
if __name__ == "__main__":
    merge_related_csvs()
