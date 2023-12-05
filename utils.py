from torchvision import transforms
import torch
import numpy as np
import av

# 预训练PyTorch模型的标准差std和均值mean
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# 视频用 从视频中提取帧
def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


# 计算并返回 y 的Gram矩阵
def gram_matrix(y):
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram


# 训练图像的转换（content
def train_transform(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform


# 风格图片的转换
def style_transform(image_size=None):
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform


# 通过mean和std反序列化图像tensors
def denormalize(tensors):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
        return tensors


# 反序列化并重新缩放图像
def deprocess(image_tensor):
    image_tensor = denormalize(image_tensor)[0]
    image_tensor *= 255
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np
