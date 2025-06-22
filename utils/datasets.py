from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, 
                 samples, 
                 indices, 
                 minority_class, 
                 basic_transform, 
                 augmented_transform,
                 is_train=True):
        self.samples = samples[indices]
        self.minority_class = minority_class
        self.basic_transform = basic_transform
        self.augmented_transform = augmented_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        label = int(label)
        image = Image.open(img_path).convert("RGB")
        
        # 학습 단계가 아니면 기본 transform 적용
        if not self.is_train:
            image = self.basic_transform(image)
        else:
            # 데이터가 적은 클래스만 증강 적용
            if label == self.minority_class:
                image = self.augmented_transform(image)
            else:
                image = self.basic_transform(image)

        return image, label