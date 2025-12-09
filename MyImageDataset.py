from torch.utils.data import Dataset
import os
from PIL import Image

class MyImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        img_dir: 图片所在的文件夹路径
        transform: 类似于 LoadCIFAR10.py 里的预处理操作
        """
        self.img_dir = img_dir
        self.transform = transform

        # 假设文件夹里是这样的：image_001.jpg, image_002.jpg ...
        # 我们用 os.listdir 获取所有文件名，并拼接成完整路径
        # self.img_info 列表里存的都是 ["路径/image_001.jpg", 标签]
        self.img_info = []

        # 模拟一个简单的加载逻辑（实际情况可能需要遍历子文件夹）
        filenames = os.listdir(img_dir)
        for name in filenames:
            # 简单粗暴的标签规则：文件名里有 'cat' 就是 0，有 'dog' 就是 1
            if 'cat' in name:
                label = 0
            elif 'dog' in name:
                label = 1
            else:
                label = -1  # 未知

            full_path = os.path.join(img_dir, name)
            self.img_info.append((full_path, label))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        def __getitem__(self, idx):
            # 1. 根据索引 idx，从列表里找到路径和标签
            img_path, label = self.img_info[idx]

            # 2. 真正读取图片 (使用 Python 的 PIL 库)
            # 这一步才是 IO 操作，最耗时
            image = Image.open(img_path).convert('RGB')

            # 3. 应用预处理 (Transform)
            # 这就是你在 LoadCIFAR10.py 里看到的 transform
            if self.transform:
                image = self.transform(image)

            # 4. 返回 (图片, 标签)
            # 这就是 Trainer.py 里 for images, labels in train_loader 拿到的东西
            return image, label