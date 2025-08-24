import os, shutil, random
from pathlib import Path

src_root = Path('all_images')  # Folder with all labeled images
dst_root = Path('data')
train_ratio = 0.8

for label_dir in src_root.iterdir():
    if not label_dir.is_dir():
        continue
    files = list(label_dir.glob('*.[jJpPnN][pPiI4]*'))  # jpg, png
    random.shuffle(files)
    cut = int(len(files) * train_ratio)
    train_files = files[:cut]
    val_files = files[cut:]

    for f in train_files:
        dest = dst_root / 'train' / label_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest / f.name)

    for f in val_files:
        dest = dst_root / 'val' / label_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest / f.name)

print('Dataset split complete.')
