import os
import pandas as pd
import requests
from urllib.parse import urlparse
import shutil


def build_dataset(data_root: str, split: str, data_src: str):
    """
    Builds the dataset from the given tsv file and saves it in the specified directory.
    :param data_root: path to save the dataset
    :param split: ['train', 'val']
    :param data_src: path to the tsv file with two columns —— caption and url
    :return: None
    """
    assert split in ['train', 'val'], f"split must be 'train' or 'val', got {split}"

    save_path = os.path.join(data_root, split)
    image_dir = os.path.join(save_path, 'images')
    os.makedirs(image_dir, exist_ok=True)

    # read the tsv file
    df = pd.read_csv(data_src, sep='\t', names=['caption', 'url'])

    # download the images and save them in the specified directory
    success_count = 0
    fail_count = 0
    for i, row in df.iterrows():
        caption = row['caption']
        url = row['url']

        file_name = f"{i:08d}.jpg"
        file_path = os.path.join(image_dir, file_name)
        if os.path.exists(file_path):
            success_count += 1
            df.iloc[i, 1] = file_path
            print(f"Downloaded {file_name}")
            continue

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"  # 模拟从谷歌搜索跳转
        }

        try:
            with requests.get(url, stream=True, headers=headers, timeout=10) as r:
                r.raise_for_status()  # 检查请求是否成功

                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

            success_count += 1
            df.iloc[i, 1] = file_path
            print(f"Downloaded {file_name}")
        except Exception as e:
            fail_count += 1
            print(f"Failed to download {file_name} (error: {str(e)})")

    df.to_csv(os.path.join(save_path, 'captions.tsv'), sep='\t', index=False, header=False)
    print(f"Downloaded {success_count} images, failed to download {fail_count} images")


if __name__ == "__main__":
    build_dataset(
        data_root=r"D:\DataSets\Img-Text\ConceptualCaptions",
        split='train',
        data_src="./Train_GCC-training.tsv"
    )
