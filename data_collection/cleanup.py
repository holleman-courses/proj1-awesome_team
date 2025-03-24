import os
import shutil

# Define the source folder (where images are currently stored)
source_folder = r"D:\School\Intro IOT\proj1-awesome_team\data_collection\no_pencil"

# Define the destination folder (where renamed images will be copied)
destination_folder = r"D:\School\Intro IOT\proj1-awesome_team\data_collection\no_pencil_clean"

# Define the label
label = "no_pencil"  # Change this to whatever label you need

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Get a sorted list of image files
image_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Iterate and copy images with new filename format
for idx, filename in enumerate(image_files):
    src_path = os.path.join(source_folder, filename)
    new_filename = f"{label}_{idx}.png"
    dst_path = os.path.join(destination_folder, new_filename)
    
    shutil.copy2(src_path, dst_path)  # Copy with metadata preserved
    print(f"Copied {src_path} -> {dst_path}")

print(f"Finished copying {len(image_files)} images to {destination_folder}")