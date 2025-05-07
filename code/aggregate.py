import pandas as pd, os
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/d/hpc/projects/FRI/ma76193"
final_csv = os.path.join(BASE_DIR, "final_scores.csv")

dfs = []
for i in range(1, 5):
    path = os.path.join(BASE_DIR, f"scores_job_{i}.csv")
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))

if not dfs:
    print("❌ No partial CSVs found.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(final_csv, index=False)
print(f"✅ Aggregated scores → {final_csv}")

# ─── PLOT: Mean Similarity per Model ───
pivot = df.pivot_table(index="model", values="mean", aggfunc='mean')
pivot.plot.barh(figsize=(6, 3), title="Mean Similarity Score per Model")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "score_plot.png"))
print("📊 Saved → score_plot.png")

# ─── PLOT: Boxplot ───
df.boxplot(column="mean", by="model", figsize=(6, 4))
plt.title("Distribution of Mean Scores by Model")
plt.suptitle("")
plt.ylabel("Similarity Score")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "score_boxplot.png"))
print("📦 Saved → score_boxplot.png")

# ─── PLOT: Heatmap ───
heat = df.pivot(index="id", columns="model", values="mean")
plt.figure(figsize=(8, 6))
plt.imshow(heat, cmap='viridis', aspect='auto')
plt.xticks(ticks=np.arange(len(heat.columns)), labels=heat.columns, rotation=45)
plt.yticks(ticks=np.arange(len(heat.index)), labels=heat.index)
plt.colorbar(label="Mean Similarity")
plt.title("Heatmap of Mean Similarity Scores")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "score_heatmap.png"))
print("🔥 Saved → score_heatmap.png")
