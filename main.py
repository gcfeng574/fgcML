import torch
from torch.autograd import Variable

from AlexNet import AlexNet
from LoadCIFAR10 import LoadCIFAR10, Construct_DataLoader
from Trainer import Trainer

# 定义参数配置信息
alexnet_config = \
{
    'num_epoch': 20,              # 训练轮次数
    'batch_size': 500,            # 每个小批量训练的样本数量
    'lr': 1e-3,                   # 学习率
    'l2_regularization':1e-4,     # L2正则化系数
    'num_classes': 10,            # 分类的类别数目
    'device_id': 0,               # 使用的GPU设备的ID号
    'use_cuda': True,             # 是否使用CUDA加速
    'model_name': './AlexNet.model' # 保存模型的文件名
}

if __name__ == "__main__":
    ####################################################################################
    # AlexNet 模型
    ####################################################################################
    train_dataset, test_dataset = LoadCIFAR10(True)
    # define AlexNet model
    alexNet = AlexNet(alexnet_config)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # # 实例化模型训练器
    trainer = Trainer(model=alexNet, config=alexnet_config)
    # # 训练
    trainer.train(train_dataset)
    # # 保存模型
    trainer.save()

    ####################################################################################
    # 模型测试阶段
    ####################################################################################
    alexNet.eval()
    alexNet.loadModel(map_location=torch.device('cpu'))
    if alexnet_config['use_cuda']:
        alexNet = alexNet.cuda()

    correct = 0
    total = 0
   # 对测试集中的每个样本进行预测，并计算出预测的精度
    for images, labels in Construct_DataLoader(test_dataset, alexnet_config['batch_size']):
        images = Variable(images)
        labels = Variable(labels)
        if alexnet_config['use_cuda']:
            images = images.cuda()
            labels = labels.cuda()

        y_pred = alexNet(images)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        temp = (predicted == labels.data).sum()
        correct += temp
    print('Accuracy of the model on the test images: %.2f%%' % (100.0 * correct / total))