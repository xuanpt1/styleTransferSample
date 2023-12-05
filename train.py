import argparse
import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from models import TransformerNet, VGG16
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
    parser.add_argument("--dataset_path", type=str, required=True, help="训练所用数据集的路径")
    parser.add_argument("--style_image", type=str, default="style-image/mosaic.jpg", help="训练用Style图片的路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="training images size")
    parser.add_argument("--style_size", type=int, help="style image size")
    parser.add_argument("--lambda_content", type=float, default=1e5, help="content loss 权重")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="style loss 权重")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--checkpoint_model", type=str, help="可选： 检查点路径")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=500, help="Batches between saving image samples")
    args = parser.parse_args()

    style_name = args.style_image.split("/")[-1].split(".")[0]
    os.makedirs(f"images/output/{style_name}-training", exist_ok=True)
    os.makedirs(f"checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练数据dataloader
    train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # 定义网络
    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    # 若有，加载检查点
    if args.checkpoint_model:
        transformer.load_state_dict(torch.load(args.checkpoint_model))

    # 定义optimizer和损失
    optimizer = Adam(transformer.parameters(), args.lr)
    l2_loss = torch.nn.MSELoss().to(device)

    # 加载风格图片
    style = style_transform(args.style_size)(Image.open(args.style_image))
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # 提取风格特征
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style]

    # 避免重新规格化，暂时每次只生成一张数据集图片的sample
    image_samples = []
    for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.jpg"), 1):
        image_samples += [style_transform(args.image_size)(Image.open(path))]
    # 直接用stack需要源图片规格一致，而MSCOCO2017训练集中图片规格不定，懒得改代码就先这样了
    image_samples = torch.stack(image_samples)

    # 评估模型并保留样本
    def save_sample(batches_done):
        transformer.eval()
        with torch.no_grad():
            output = transformer(image_samples.to(device))
        image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid, f"images/outputs/{style_name}-training/{batches_done}.jpg", nrow=4)
        transformer.train()

    for epoch in range(args.epochs):
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()

            images_original = images.to(device)
            images_transformed = transformer(images_original)

            # 提取特征
            features_original = vgg(images_original)
            features_transformed = vgg(images_transformed)

            # 计算内容损失
            content_loss = args.lambda_content * l2_loss(features_transformed.relu2_2, features_original.relu2_2)

            # 计算风格损失
            style_loss = 0
            for ft_y, gm_s in zip(features_transformed, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += l2_loss(gm_y, gm_s[: images.size(0), :, :])
            style_loss *= args.lambda_style

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["total"] += [total_loss.item()]

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    args.epochs,
                    batch_i,
                    len(train_dataset),
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

            batches_done = epoch * len(dataloader) + batch_i + 1
            if batches_done % args.sample_interval == 0:
                save_sample(batches_done)

            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                torch.save(transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")
