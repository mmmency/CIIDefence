"""
2023 4 16
bsxm

训练：先生成类激活图，然后对前n个类激活图找前p个显著点，生成(2w+1)*(2w+1)的mask，inpaint，再fusion
用VGG-16来生成类激活图和预测
1. 500training set 选择参数，3000验证集分析实验分类性能
2. 将3500张图片先用VGG预测，只保留预测正确的图像
3. 对500张图片生成FGSM [9]， IGSM [13]， DFool [19]， PGD[18]和C&W[5]攻击的对抗样本
"""

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:7")  # 使用第一块GPU设备
    torch.cuda.set_device(device)  # 设置当前设备
    print('Using GPU:', torch.cuda.get_device_name(device),
          "当前选中的显卡：", torch.cuda.current_device())
else:
    device = torch.device("cpu")
    print('Using CPU')

from torchvision import models,transforms,utils
from PIL import Image
from torch.nn import functional as F
import re
import os
use_cuda = True  # 是否使用GPU


import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchattacks


os.chdir(sys.path[0])
class_path = 'imagenet_classes.txt'
image_path = '/home/mxma/Project/CIIDefense/imagenet-sample-images'
#image_path = '/home/mxma/Project/CIIDefense/test_data'
cam_path = "./genImg/clean/cam_img"
mask_path = "./genImg/clean/mask_img"
inpaint_path = "./genImg/clean/inpaint_img"
diff_path = "./genImg/clean/diff_img"

adv_path = "./genImg/adv/adv_img"
adv_cam_path = "./genImg/adv/adv_cam_img"
adv_mask_path = "./genImg/adv/adv_mask_img"
adv_inpaint_path = "./genImg/adv/adv_inpaint_img"
adv_diff_path = "./genImg/adv/adv_diff_img"

folder_path = [cam_path,mask_path,inpaint_path,diff_path]
adv_folder_path = [adv_path,adv_cam_path,adv_mask_path,adv_inpaint_path,adv_diff_path]
for folder in folder_path+adv_folder_path:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Folder created:", folder)
    #else:
    #    print("Folder already exists:", folder)

net_id = 3
n_class = 1#5     #生成n个类别的激活图
p_class = 3     #每张激活图选择top p个mask区域
w_masksize = 3  #对p个点，分别取(2w+1)*(2w+1)个pixel的mask
epsilon = 0.01
is_adv = True

attack_mode = 2     #1——FGSM，2——CW


if net_id in [2,6]:
    finalconv_name = 'layer4'
elif net_id == 5:
    finalconv_name = 'Mixed_7c'
else:
    finalconv_name = 'features'


def load_class():
    with open(class_path) as f:
        classes = [s.strip() for s in f.readlines()]
    return classes

#加载预训练网络
def Net():
    if net_id == 1:
        return models.squeezenet1_1(pretrained = True)
    elif net_id == 2:
        return models.resnet18(pretrained = True)
    elif net_id == 3:
        return models.densenet161(pretrained = True)
    elif net_id == 4:
        return models.vgg16(pretrained = True)
    elif net_id == 5:
        return models.inception_v3(pretrained = True)
    elif net_id == 6:
        return models.resnet101(pretrained = True)


classes = load_class()

net = Net().to(device)

class ImageData(torch.utils.data.Dataset):
    def __init__(self,dataset_dir):
        self.all_image_paths = []
        for x in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir,x)
            if self.__is_image_file(x) and self.__is_RGB_file(path):
                self.all_image_paths.append(path)

    def __is_image_file(self , filename):
        return any(filename.endswith(extension)
                    for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def __is_RGB_file(self,filename):
        img = Image.open(filename)
        return img.mode == 'RGB'

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self , index):
        input_image = Image.open(self.all_image_paths[index])

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(input_image).to(device)

        #正则表达式找到文件名中的label, 到txt文件中找index
        file_name = re.search(r'n[0-9]{8}.+\.', self.all_image_paths[index]).group()
        class_name = file_name[10:-1].replace('_', ' ')
        #with open("imagenet_classes.txt", "r") as f:
        #    categories = [s.strip() for s in f.readlines()]
        label = classes.index(class_name)
        if is_adv :
            if attack_mode ==1:
                img.requires_grad = True
                output = net(img.unsqueeze(0))
                target = torch.tensor([label]).to(device)
                #loss = F.cross_entropy(output, torch.argmax(output))
                loss = F.nll_loss(output, torch.tensor([target]).to(device))

                loss.backward()
                data_grad = img.grad.data
                perturbed_image = fgsm_attack(img, epsilon, data_grad)
            else:
                preprocess_cw = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),])
                cw_img = preprocess_cw(input_image)
                cw_attack = torchattacks.CW(net,c=1, kappa=0,lr = 0.01)
                perturbed_image = cw_attack(cw_img.unsqueeze(0),torch.tensor(label)).squeeze(0)


        else:
            perturbed_image = img

        return img.detach(),label,perturbed_image.detach()



#加载数据
def Data():
    data = ImageData(image_path)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=1,
                                              shuffle=True)
    return data_loader

data_loader = Data()
print("read data from:",image_path)
print("使用网络：",type(net).__name__)
net.eval()




def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

features_blobs = []     # 后面用于存放特征图

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# 获取 features 模块的输出
'''
Hook函数`hook_feature`绑定到最终的卷积层上，以便在网络进行前向传递时自动获取最终卷积层的输出特征图。
最终，在整个网络进行前向传递时，每次经过最终卷积层时，都会调用一次`hook_feature`函数，
将最终卷积层的输出添加到`features_blobs`列表中，
最后返回的`features_blobs`列表就包含了整个输入图像在最后一层卷积计算后得到的所有特征图。
'''

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
if net_id == 4:
    weight_softmax = np.dot(np.dot(np.squeeze(params[-2].cpu().data.numpy()), \
    np.squeeze(params[-4].cpu().data.numpy())), np.squeeze(params[-6].cpu().data.numpy()))
else:
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    


def create_CAM(feature_conv, weight_softmax, class_idx,label):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    output_mask = []
    i = 0
    for idx in class_idx:
        # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，reshape了map的形状)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        #归一化
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        #生成mask
        mask_img = np.uint8(255 * create_mask(cam_img))
        output_mask.append(mask_img)

        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        output_cam.append(cam_img)

        camImg_name = classes[label.item()] + str(i) + '.png'
        mask_name = classes[label.item()] + str(i) + '_mask.png'
        if is_adv :
            cv2.imwrite(os.path.join(adv_cam_path, camImg_name), cam_img)
            cv2.imwrite(os.path.join(adv_mask_path, mask_name), mask_img)
        else:
            cv2.imwrite(os.path.join(cam_path, camImg_name), cam_img)
            cv2.imwrite(os.path.join(mask_path, mask_name), mask_img)

        i = i+1


    return output_cam,output_mask


def create_mask(cam_img):
    #输入一个大小为13*13的类激活图(归一化的)，转换为一个256*256的mask
    max_values = np.sort(cam_img.flatten())[-p_class:]
    size_upsample = (256, 256)
    cam_img = cv2.resize(cam_img, size_upsample)

    p = p_class
    mask = np.zeros(size_upsample)
    while p:
        max_val = np.max(cam_img)
        max_index = np.where(cam_img == max_val)  # max_num对应在图上的坐标，可能有很多
        if len(max_index[0]) > 1:
            center = np.mean(max_index, axis=1).astype(int)
        else:
            center = max_index[0][0], max_index[1][0]
        cam_img[max(center[0] - 100, 0):min(center[0] + 1 + 100, 256), max(center[1] - 100, 0):min(center[1] + 1 + 100, 256)] = 0
        mask[center[0], center[1]] = 0

        # 在数组中将最大值的位置处和周围的7x7个位置的值设为1——需要修复
        mask[max(center[0] - p_class, 0):min(center[0] + 1 + p_class, 256),
        max(center[1] - p_class, 0):min(center[1] + 1 + p_class, 256)] = 1

        p = p-1

    return mask



def de_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # mean和std分别为均值和标准差，要分别对每个通道进行反标准化
    #tensor = tensor.clone()
    for channel in range(tensor.size(0)):
        tensor[channel].mul_(std[channel]).add_(mean[channel])
    return tensor

def clip_pixel_value(tensor, min_value=0, max_value=1):
    # 将张量中超出指定取值范围的像素值进行剪裁,默认剪裁到[0,255]
    tensor = torch.clamp(tensor, min_value, max_value)
    return tensor




def create_inpaintImg(img,MASKs,lable):
    # 将Tensor反标准化、裁剪，squeeze多余通道，恢复到0-255的图片，再转换为numpy数组
    denorm_img = clip_pixel_value(de_normalize(img.squeeze()))
    data_np = (255.0 * denorm_img).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    mask = MASKs[0][:224, :224].astype(np.uint8)
    # cv2.inpaint函数是OpenCV中的图像修复函数，输入是一个NumPy数组或OpenCV中的Mat对象
    dst = cv2.inpaint(data_np, mask, 3, cv2.INPAINT_TELEA)

    # 保存生成的inpaint图片
    img_inp = np.uint8(np.clip(dst, 0, 255))
    inpaint_name = classes[lable.item()] + '_inp.png'
    if is_adv:
        cv2.imwrite(os.path.join(adv_inpaint_path, inpaint_name), cv2.cvtColor(img_inp, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(os.path.join(inpaint_path, inpaint_name), cv2.cvtColor(img_inp, cv2.COLOR_BGR2RGB))

    # 把生成的numpy数组转换为tensor用于分类网络
    img_inpaint = torch.from_numpy(dst.astype(np.float32).transpose((2, 0, 1))).to(device)
    # preprocess = transforms.Compose([
    #    transforms.Resize(256),
    #    transforms.CenterCrop(224),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化影响可视化效果 屏蔽
    # ])
    img_inpaint = torch.unsqueeze(img_inpaint, 0)
    # data_inpaint = (data_inpaint - torch.min(data_inpaint)) / (torch.max(data_inpaint) - torch.min(data_inpaint))
    img_inpaint = torch.div(img_inpaint, img_inpaint.max())

    # 计算差异
    diff = torch.abs(img.squeeze() - img_inpaint.squeeze())
    diff_gray = transforms.Grayscale(num_output_channels=1)(diff)
    diff_name = classes[lable.item()] + '_diff.png'
    if is_adv:
        utils.save_image(diff_gray, os.path.join(adv_diff_path, diff_name), normalize=True)
    else:
        utils.save_image(diff_gray, os.path.join(diff_path, diff_name), normalize=True)
    return img_inpaint



def CAM(net,data_loader):

    predict_correct_count = 0
    predict_inpaint_count = 0
    test_count = len(data_loader)


    if is_adv:
        print("data为对抗样本")
    else:
        print("data为干净图像")

    for data,label,adv_data in data_loader:
        # 移到显卡 Send the data and label to the device
        #data, lable,adv_img = data.to(device), lable.to(device),adv_img.to(device)
        label,adv_data = label.to(device),adv_data.to(device)


        #保存对抗样本
        adv_name = classes[label.item()]  + '_adv.png'
        adv_img = adv_data.squeeze(0)
        adv_img = transforms.functional.to_pil_image(adv_img)
        adv_img.save(os.path.join(adv_path,adv_name))
        """
        以下生成CAM、Mask、Inpainting图片的base是否为对抗样本
        """

        if is_adv:
            test_data = adv_data
        else:
            test_data = data


        #生成网络预测结果
        logit = net(test_data)
        logit_norm = F.softmax(logit, dim=1).data.squeeze()  # [1,1000] => 1000
        max_probs, max_idx = torch.max(logit_norm, dim=0)
        #probs = probs.cpu().numpy()
        #idx = idx.cpu().numpy()

        # 保存并输出前n个预测结果
        #for i in range(0, n_class):
        #    print('predict result:\n{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        # clean预测正确
        if max_idx.item() == label.item():
            predict_correct_count = predict_correct_count+1
        top_n_values, top_n_indices = torch.topk(logit_norm, k=n_class)
        predict_class = top_n_indices.tolist()

        CAMs,MASKs = create_CAM(features_blobs[0], weight_softmax, predict_class,label)

        """
        #保存类激活图和每个类的mask
        for i in range(0, n_class):
            camImg_name = classes[label.item()] + str(i) + '.png'
            mask_name =  classes[label.item()] + str(i) + '_mask.png'
            cv2.imwrite(os.path.join(cam_path,camImg_name), CAMs[i])
            cv2.imwrite(os.path.join(mask_path,mask_name), MASKs[i])
        """

        #生成inpaint的tensor
        data_inpaint = create_inpaintImg(data,MASKs,label)
        nor_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data_inpaint = nor_(data_inpaint)
        logit_inp = net(data_inpaint)

        logit_inp_norm = F.softmax(logit_inp, dim=1).data.squeeze()
        max_probs_inp, max_idx_inp = torch.max(logit_inp_norm, dim=0)

        # 预测正确
        if max_idx_inp.item() == label.item():
            predict_inpaint_count = predict_inpaint_count + 1

    accuracy_clean = predict_correct_count / float(test_count)
    accuracy_inpaint = predict_inpaint_count / float(test_count)
    print("Clean Test Accuracy \t= {} / {} = {}".format(predict_correct_count, test_count, accuracy_clean))
    print("Inpaint Test Accuracy \t= {} / {} = {}".format(predict_inpaint_count, test_count, accuracy_inpaint))

    #return accuracy_clean,accuracy_inpaint

CAM(net,data_loader)








