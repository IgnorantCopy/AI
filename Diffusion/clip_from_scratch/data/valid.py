import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=data_root, type=str, help="Path to the dataset root.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args["data_root"]
    train_image_paths = glob.glob(os.path.join(data_root, "train/images/*.jpg"))
    val_image_paths = glob.glob(os.path.join(data_root, "val/images/*.jpg"))
    image_paths = train_image_paths + val_image_paths
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert("RGB")
        h, w = image.size
        if h < 100 or w < 100:
            caption_path = image_path.replace("images", "captions").replace(".jpg", ".txt")
            metadata = image_path.replace("images", "metadata").replace(".jpg", ".json")
            os.remove(image_path)
            os.remove(caption_path)
            os.remove(metadata)
            print(f"Removed {image_path}")


if __name__ == '__main__':
    main()