import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from Image_feat_model import PIC_model
from PIL import Image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def extract_image_features(image_path, model):
    transform = transforms.Compose([
        transforms.Resize(224),             # 将图像大小调整为224x224
        transforms.CenterCrop(224),         # 在中心裁剪图像
        transforms.ToTensor(),              # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])
    
    image = load_image(image_path, transform)
    image_features = model.features(image)
    img_features = image_features.view(image_features.size(0), -1)  # 展平特征张量
    return img_features

# # 使用预训练的VGG16模型
# model = models.vgg16(pretrained=True)
# model = model.features
# model.eval()  # 设为评估模式

model_path = 'weights/vgg16-397923af.pth'
model = PIC_model()
model.load_state_dict(torch.load(model_path))
model.eval()


data_pic_dir = 'data_pic/Flicker8k_Dataset'
output_dir = 'utils_data/image_features'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open('data_text/Flickr_8k.testImages.txt') as f:
# with open('test.txt') as f:
    text = f.read()
    image_files = text.split('\n')

n = len(image_files)
for i in range(len(image_files)):
    image_file = image_files[i]
    image_path = os.path.join(data_pic_dir, image_file)
    image_features = extract_image_features(image_path, model)

    output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + '.npy')
    np.save(output_path, image_features.detach().numpy())

    if i % 200 == 0:
        print(f'{i}/{n}')
    # 保存图像特征为.npy文件


print("Image feature extraction complete.")
# with open('utils_data/pic_feature.pkl','rb') as pickle_file:
#     x = pickle.load(pickle_file)
#     print(pic_data)
#     print(x)
#     print(len(x))