import os
import glob
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import plotly.graph_objects as go


def calc_ratio(pred_dir: str, gt_dir: str, prefix: str = None, postfix: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    to_tensor = ToTensor()
    gt_images = glob.glob(os.path.join(gt_dir, "*"))
    ratios = []
    for image_path in gt_images:
        image_name = os.path.basename(image_path).split(".")
        if prefix is not None:
            image_name[0] = prefix + image_name[0]
        if postfix is not None:
            image_name[0] = image_name[0] + postfix

        pred_image_path = os.path.join(pred_dir, image_name[0] + '.' + image_name[1])
        gt = to_tensor(Image.open(image_path)).to(device)
        pred = to_tensor(Image.open(pred_image_path)).to(device)
        ratio = (gt.mean() / pred.mean()).item()
        ratios.append(ratio)
    return ratios


if __name__ == '__main__':
    lolv1_ratios = calc_ratio("D:/DataSets/LLIE/LOLv1/our485/low", "D:/DataSets/LLIE/LOLv1/our485/high")
    lolv2_real_ratios = calc_ratio("D:/DataSets/LLIE/LOLv2/Real_captured/Train/Low", "D:/DataSets/LLIE/LOLv2/Real_captured/Train/Normal")
    lolv2_syn_ratios = calc_ratio("D:/DataSets/LLIE/LOLv2/Synthetic/Train/Low", "D:/DataSets/LLIE/LOLv2/Synthetic/Train/Normal")

    fig = go.Figure(data=[
        go.Histogram(x=lolv1_ratios, nbinsx=50, name='LOLv1', opacity=0.5, histnorm="probability"),
        go.Histogram(x=lolv2_real_ratios, nbinsx=100, name='LOLv2-Real', opacity=0.5, histnorm="probability"),
        go.Histogram(x=lolv2_syn_ratios, nbinsx=20, name='LOLv2-Synthetic', opacity=0.5, histnorm="probability"),
        # go.Histogram(x=lolv1_ratios + lolv2_real_ratios + lolv2_syn_ratios, nbinsx=100, name='All', opacity=0.5, histnorm="probability"),
    ])
    fig.update_layout(
        barmode='overlay',
        xaxis_title='Illumination Gain',
        yaxis_title='Probability',
        legend=dict(
            orientation="v",  # 垂直排列
            yanchor="top",  # 图例顶部对齐
            y=0.99,  # 纵向位置（靠近顶部）
            xanchor="right",  # 图例右侧对齐
            x=0.99,  # 横向位置（靠近右侧）
            bgcolor="white",  # 背景白色（更清晰）
            bordercolor="black",  # 边框（可选）
            borderwidth=1  # 边框宽度（可选）
        )
    )
    fig.show()
