import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 定义构造数据加载器的函数
def Construct_DataLoader(dataset, batchsize):
    return DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载CIFAR-10数据集函数
def LoadCIFAR10(download=False):
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, transform=transform, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, transform=transform)
    return train_dataset, test_dataset