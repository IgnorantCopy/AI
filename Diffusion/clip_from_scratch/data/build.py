import os
import shutil
import glob
from typing import List
from sklearn.model_selection import train_test_split


"""
img2dataset --url_list ConceptualCaptions --input_format "tsv" --url_col "URL" --caption_col "TEXT"
            --output_format files --output_folder ConceptualCaptionsData --processes_count 16
            --thread_count 64 --image_size 384 --resize_only_if_bigger=True --resize_mode="keep_ratio"
            --skip_reencode=True --enable_wandb True
"""

def reorganize_dataset(data_root: str, exclude_dirs: List[str], val_ratio: float = 0.2, seed: int = 42):
    data_dirs = glob.glob(os.path.join(data_root, '*'))
    image_paths = []
    caption_paths = []
    metadata_paths = []
    for data_dir in data_dirs:
        if data_dir in exclude_dirs:
            continue
        images = glob.glob(os.path.join(data_dir, '*.jpg'))
        captions = glob.glob(os.path.join(data_dir, '*.txt'))
        metadata = glob.glob(os.path.join(data_dir, '*.json'))
        images.sort()
        captions.sort()
        metadata.sort()
        image_paths.extend(images)
        caption_paths.extend(captions)
        metadata_paths.extend(metadata)

    totals = len(image_paths)
    train_size = int(totals * (1 - val_ratio))
    train_image_paths, val_image_paths, train_caption_paths, val_caption_paths, train_metadata_paths, val_metadata_paths = \
        train_test_split(image_paths, caption_paths, metadata_paths, train_size=train_size, random_state=seed)
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    train_image_dir = os.path.join(train_dir, 'images')
    val_image_dir = os.path.join(val_dir, 'images')
    train_caption_dir = os.path.join(train_dir, 'captions')
    val_caption_dir = os.path.join(val_dir, 'captions')
    train_metadata_dir = os.path.join(train_dir,'metadata')
    val_metadata_dir = os.path.join(val_dir,'metadata')
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_caption_dir, exist_ok=True)
    os.makedirs(val_caption_dir, exist_ok=True)
    os.makedirs(train_metadata_dir, exist_ok=True)
    os.makedirs(val_metadata_dir, exist_ok=True)

    for i, (image_path, caption_path, metadata_path) in enumerate(zip(train_image_paths, train_caption_paths, train_metadata_paths)):
        new_image_path = os.path.join(train_image_dir, os.path.basename(image_path))
        new_caption_path = os.path.join(train_caption_dir, os.path.basename(caption_path))
        new_metadata_path = os.path.join(train_metadata_dir, os.path.basename(metadata_path))
        shutil.copyfile(image_path, new_image_path)
        shutil.copyfile(caption_path, new_caption_path)
        shutil.copyfile(metadata_path, new_metadata_path)

    for i, (image_path, caption_path, metadata_path) in enumerate(zip(val_image_paths, val_caption_paths, val_metadata_paths)):
        new_image_path = os.path.join(val_image_dir, os.path.basename(image_path))
        new_caption_path = os.path.join(val_caption_dir, os.path.basename(caption_path))
        new_metadata_path = os.path.join(val_metadata_dir, os.path.basename(metadata_path))
        shutil.copyfile(image_path, new_image_path)
        shutil.copyfile(caption_path, new_caption_path)
        shutil.copyfile(metadata_path, new_metadata_path)


if __name__ == "__main__":
    reorganize_dataset(r"D:\DataSets\Img-Text\ConceptualCaptionsData", exclude_dirs=["_tmp"])
