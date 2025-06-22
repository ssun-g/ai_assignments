import random

import numpy as np

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import matplotlib.pyplot as plt


def visualization_per_class(model, dataset, target_layer, transform, mean, std):
    # 클래스 별 인덱스 저장
    cls_indices = {i: [] for i in range(len(dataset.classes))}
    for i, (_, label) in enumerate(dataset.samples):
        cls_indices[label].append(i)

    # 각 클래스에서 랜덤 샘플 1개씩 선택
    selected = {}
    for label, i in cls_indices.items():
        selected[label] = random.choice(i)

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 이미지 시각화
    plt.figure(figsize=(15, 5))

    for i, (label, idx) in enumerate(selected.items()):
        img_path, _ = dataset.samples[idx]
        img = Image.open(img_path).convert('RGB')

        # transform 적용
        img_tensor = transform(img)
        input_tensor = img_tensor.unsqueeze(0)

        outputs, _ = model(input_tensor)
        pred = outputs.argmax(dim=1).item()
        targets = [ClassifierOutputTarget(pred)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # 이미지 역정규화
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        plt.subplot(1, len(selected), i+1)
        plt.imshow(visualization)
        plt.title(f"{dataset.classes[label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()