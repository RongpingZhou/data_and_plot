from tbparse import SummaryReader
import pandas as pd
import os

# log_dir = "/home/wsl/mygit/rwrl_backup/runs/AtlantisNoFrameskip-v4/DQN_1/events.out.tfevents.1773319891.A512025.293052.0"
# output_dir = "/home/wsl/mygit/rwrl_backup/runs/AtlantisNoFrameskip-v4/DQN_1/"
log_dir = "C:\\GitHub\\data_and_plot\\training\\real_world\\events.out.tfevents.1766320106.25845323626c.462.0"
output_dir = "C:\\GitHub\\data_and_plot\\training\\real_world\\"
os.makedirs(output_dir, exist_ok=True)

reader = SummaryReader(log_dir, extra_columns={'wall_time'})

# Export scalars
df = reader.scalars
df.to_csv(os.path.join(output_dir, "scalars.csv"), index=False)

# for tag, group in df.groupby("tag"):
#     safe_tag = tag.replace("/", "_")

#     # Reorder columns: tag first, then the rest
#     cols = ["tag"] + [c for c in group.columns if c != "tag"]
#     print(f"Exporting tag: {tag} to {safe_tag}.csv with columns: {cols}")
#     group = group[cols]

#     out_path = os.path.join(output_dir, f"{safe_tag}.csv")
#     group.to_csv(out_path, index=False)

for tag, group in df.groupby("tag"):
    safe_tag = tag.replace("/", "_")

    # Drop tag and dir_name columns if they exist
    group = group.drop(columns=["tag"], errors="ignore")

    out_path = os.path.join(output_dir, f"{safe_tag}.csv")
    group.to_csv(out_path, index=False)


# Export histograms
if reader.histograms is not None:
    reader.histograms.to_csv(os.path.join(output_dir, "histograms.csv"), index=False)

# Export tensors
if reader.tensors is not None:
    reader.tensors.to_csv(os.path.join(output_dir, "tensors.csv"), index=False)

# Export text
if reader.text is not None:
    reader.text.to_csv(os.path.join(output_dir, "text.csv"), index=False)

# Export images metadata (not the raw images)
if reader.images is not None:
    reader.images.to_csv(os.path.join(output_dir, "images.csv"), index=False)

print("All TensorBoard data exported to CSV.")