import json
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
#import tensorflow_hub as hub
import re
import os
import sys

class_path = 'imagenet_classes.txt'
image_path = '/home/mxma/Project/CIIDefense/inpaint_img'
    #'/home/mxma/Data/images'

def load_class():
    with open(class_path) as f:
        classes = [s.strip() for s in f.readlines()]
    return classes

classes = load_class()

def Net():


    # 加载EfficientNet-L2模型
    #return hub.KerasLayer('https://tfhub.dev/google/efficientnet/l2/feature-vector/1')
    return models.googlenet(pretrained=True)
    #return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    #return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt')


def Data():
    """
    Build a Dataset with 1000 images, one per image-net class.
    Clone this ↓ repo and move on.
    Ref: https://github.com/EliSchwartz/imagenet-sample-images
    """
    imagenet_data = ImageNet_sample(image_path)  # 数据集路径
    test_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=1,
                                              shuffle=True)
    return test_loader


class ImageNet_sample(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.all_image_paths = []  # 收集有效图片
        for x in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, x)
            if self.__is_image_file(x) and self.__is_RGB_file(path):  # 过滤灰度图、非图片文件
                self.all_image_paths.append(path)

    def __is_image_file(self, filename):
        """
        判断文件是否是图片
        """
        return any(filename.endswith(extension)
                   for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def __is_RGB_file(self, filename):
        """
        判断文件是否是RGB图片
        """
        img = Image.open(filename)
        return img.mode == 'RGB'

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        # 图形预处理
        input_image = Image.open(self.all_image_paths[index])
        preprocess = transforms.Compose([

            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化影响可视化效果 屏蔽
        ])
        img = preprocess(input_image)

        # 正则表达式找到文件名中的label, 到txt文件中找index
        #file_name = re.search(r'n[0-9]{8}.+\.', self.all_image_paths[index]).group()
        #class_name = file_name[10:-1].replace('_', ' ')

        file_name = os.path.basename(self.all_image_paths[index])
        class_name = file_name[:-8]

        # with open("imagenet_classes.txt", "r") as f:
        #    categories = [s.strip() for s in f.readlines()]
        label = classes.index(class_name)


        return img, label


test_loader = Data()
model = Net()


# print("test_loader",len(test_loader.dataset))
# print(model)


def test(model, test_loader, test_count=10000):
    """
    model, test_loader, device: 模型、数据集和设备
    epsilon: 扰动的大小
    test_count: 测试图片数量
    """
    # Accuracy counter
    correct = 0
    # adversarial_examples = []
    tested_count = 0
    # print("test_loader",len(test_loader.dataset),"test_count",test_count)

    # 遍历 Loop over all examples in test set
    for data, target in test_loader:

        # 正向计算拿到预测 Forward pass the data through the model
        output = model(data)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if pred.item() == target.item():
            correct += 1

        tested_count += 1
        # if tested_count >= test_count:
        #    break
    # 计算正确率 Calculate final accuracy for this epsilon
    accuracy = correct / float(tested_count)
    print("Test Accuracy = {} / {} = {}".format(correct, tested_count, accuracy))

    # Return the accuracy and an adversarial example
    return accuracy


accuracies = []

# Run test for each epsilon

acc = test(model, test_loader, test_count=1000)
accuracies.append(acc)







