""" Builds the ImageFolder dataset (benign / malignant) from the CBIS-DDSM CSVs and JPEG folders.
    inputs: mass and calc train-set CSVs in data/csv, JPEG series folders in data/jpeg
    outputs: data/clean_training_set/{benign,malignant}/
"""
import os
import shutil
import pandas as pd

# local paths, update for the local machine
base_dir = "data"
csv_dir = os.path.join(base_dir, "csv")
jpeg_dir = os.path.join(base_dir, "jpeg")
output_dir = "clean_training_set"

print("[SYSTEM] Initiating target folder reorganization...")

# 1. load the metadata CSVs (mass and calcification cases)
sheets = []
for file_name in ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv"]:
    path = os.path.join(csv_dir, file_name)
    if os.path.exists(path):
        sheets.append(pd.read_csv(path))
        print(f"[LOAD] Indexed metadata reference: {file_name}")

if not sheets:
    print("[CRITICAL] Index metadata references missing from target paths.")
    exit(1)

metadata = pd.concat(sheets, ignore_index=True)

# 2. create the class folders
for cls in ["benign", "malignant"]:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# 3. copy each image into its class folder
copied_count = 0
print("[MIGRATION] Mapping path strings directly to physical hardware storage blocks...")

for _, row in metadata.iterrows():
    # MALIGNANT -> malignant, everything else (incl. without callback) -> benign
    pathology = str(row['pathology']).upper()
    target_class = "malignant" if "MALIGNANT" in pathology else "benign"

    # CSV path holds study and series UIDs; the last UID is the JPEG folder name
    raw_path_str = str(row['image file path'])

    # normalise slashes and split into parts
    path_parts = raw_path_str.replace('/', os.sep).replace('\\', os.sep).split(os.sep)

    # walk backwards to the nearest UID folder
    uid_folder = None
    for part in reversed(path_parts):
        if part.startswith("1.3.6."):
            uid_folder = part
            break

    if uid_folder:
        source_folder = os.path.join(jpeg_dir, uid_folder)

        if os.path.isdir(source_folder):
            images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in images:
                src_file_path = os.path.join(source_folder, img_file)
                # UID prefix keeps file names unique
                new_name = f"{uid_folder}_{img_file}"
                dest_file_path = os.path.join(output_dir, target_class, new_name)

                # skip files already copied
                if not os.path.exists(dest_file_path):
                    shutil.copy2(src_file_path, dest_file_path)
                    copied_count += 1

print(f"\n[SUCCESS] Migration script completed execution.")
print(f" -> Production dataset path established: {os.path.abspath(output_dir)}")
print(f" -> Total clinical matrices successfully structured: {copied_count}")
