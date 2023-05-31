"""
生成类激活图

"""
import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

# input image
LABELS_file = 'imagenet_classes.txt'
#image_file = 'test_data/n02391049_zebra.JPEG' 
image_file = 'test_data/n01443537_goldfish.JPEG'
n_class = 5     #生成n个类别的激活图
p_class = 5#3     #每张激活图选择top p个mask区域
w_masksize = 100#3  #对p个点，分别取(2w+1)*(2w+1)个pixel的mask

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 3
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
elif model_id == 4:
    net = models.vgg16(pretrained=True)
    finalconv_name = 'features'



net.eval()

# 获得特定层的feature map
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
#-2表示classifier.1的权重。因此，我们要的就是索引为-2的参数。
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    mask_img = []
    for idx in class_idx:
        # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，reshape了map的形状)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        #归一化
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        # 生成mask
        mask_img.append(np.uint8(255 * creat_mask(cam_img)))

        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam,mask_img


def creat_mask(cam_img):
    # 输入一个大小为13*13的类激活图(归一化的)，转换为一个256*256的mask
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
        cam_img[max(center[0] - 100, 0):min(center[0] + 1 + 100, 256),
        max(center[1] - 100, 0):min(center[1] + 1 + 100, 256)] = 0
        mask[center[0], center[1]] = 0

        # 在数组中将最大值的位置处和周围的7x7个位置的值设为1——需要修复
        mask[max(center[0] - p_class, 0):min(center[0] + 1 + p_class, 256),
        max(center[1] - p_class, 0):min(center[1] + 1 + p_class, 256)] = 1

        p = p - 1
    return mask


#ImageNet数据集均值以及标准差
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# load the imagenet category list
with open(LABELS_file) as f:
    classes = [s.strip() for s in f.readlines()]



h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()


# output the prediction
for i in range(0, 5):
    print('predict result:\n{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs,Mask = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
cv2.imwrite('CAM.jpg', CAMs[0])
cv2.imwrite('mask.jpg',Mask[0])















# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
cam_merge = heatmap * 0.5 + img * 0.3
mask = cv2.cvtColor(cv2.resize(Mask[0],(width, height)), cv2.COLOR_GRAY2BGR)
mask_merge = np.where(mask > 0, mask, img)

cv2.imwrite('CAM_merge.jpg', cam_merge)
cv2.imwrite('Mask_merge.jpg',mask_merge)



"""
"""


