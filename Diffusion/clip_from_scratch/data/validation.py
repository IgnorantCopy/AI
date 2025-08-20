import os
import pandas as pd
from PIL import Image, UnidentifiedImageError


def is_url(url: str) -> bool:
    return url.startswith('http')


def is_jpeg(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except UnidentifiedImageError:
        return False


def main(file_path: str):
    df = pd.read_csv(file_path, sep='\t', names=['caption', 'path'])
    url_mask = df['path'].apply(is_url)
    df = df[~url_mask]
    jpeg_mask = df['path'].apply(is_jpeg)
    df = df[jpeg_mask]
    save_path = os.path.join(os.path.dirname(file_path), 'filtered_captions.tsv')
    df.to_csv(save_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    main(r"D:\DataSets\Img-Text\ConceptualCaptions\val\captions.tsv")