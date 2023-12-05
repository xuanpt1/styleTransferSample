from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    args = parser.parse_args()
    print(args)

    os.makedirs("images/outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = style_transform()

    # 加载checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(args.checkpoint_model))
    transformer.eval()

    # 处理输入图片
    image_tensor = Variable(transform(Image.open(args.image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # 保存图片
    fn = args.image_path.split("/")[-1]
    save_image(stylized_image, f"images/outputs/stylized-{fn}")
