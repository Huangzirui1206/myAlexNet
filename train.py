#train.py

from dataclasses import make_dataclass
from msilib import sequence
from random import sample, shuffle
from matplotlib.hatch import SmallCircles
#from matplotlib import image
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time
from shutil import copy, rmtree
from PIL import Image 
from torch.utils.data.sampler import SequentialSampler

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.data_info, self.class_to_idx = self.get_img_info(root)         
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)
    
    def get_img_info(data_dir)->tuple[list[tuple[str,int]], dict[str, int]]:
        data_info = list()
        _, class_to_idx = SequenceDataset.find_classes(data_dir)
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = class_to_idx[sub_dir]
                    data_info.append((path_img, int(label)))
        #Using sorted listed samples
        data_info = sorted(data_info, key = lambda t: t[0])
        return data_info, class_to_idx
    
    @staticmethod
    def find_classes(directory: str) -> tuple[list[str], dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

#data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = os.getcwd()
image_path = data_root + "/flower_data/"  # flower data set path

#construct sequence
train_data_path = image_path + "/train/"
if os.path.exists(train_data_path + '/data_sequence/'):
    rmtree(train_data_path + '/data_sequence/')
sample_class = [cla for cla in os.listdir(train_data_path) if ".txt" not in cla]
mkfile(train_data_path + '/data_sequence/')
for cla in sample_class:
    mkfile(train_data_path + '/data_sequence/'+cla)

rename_cnt = 0
for cla in sample_class:
    cla_path = train_data_path + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    rename_cnt += num
rename_list = [i for i in range(rename_cnt)]
shuffle(rename_list)
rename_base = 0
for cla in sample_class:
    cla_path = train_data_path + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    for index, image in enumerate(images):
        old_path = cla_path + image
        new_path = train_data_path + '/data_sequence/' + cla
        copy(old_path, os.path.join(new_path, str(rename_list[index + rename_base]).zfill(6) + '.jpg'))
        print("\r[{}] Training processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    rename_base += num
    print()
    
print("Training processing done!")
train_data_path += '/data_sequence'
print(train_data_path)

#train_dataset = datasets.ImageFolder(root=train_data_path,
#                                     transform=data_transform["train"])
#To simulate a list, using sequenceDataset
train_dataset = datasets.ImageFolder(root=train_data_path,
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

#shuffle epoch
#batch_size = 32
#To simulate continuous learning, set batch_size = 1
batch_size = 1

# Shuffle the samples
#train_loader = torch.utils.data.DataLoader(train_dataset,
#                                           batch_size=batch_size, shuffle=True,
#                                           num_workers=0)
# Do not shuffle
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.next()
#print(test_image[0].size(),type(test_image[0]))
#print(test_label[0],test_label[0].item(),type(test_label[0]))


#显示图像，之前需把validate_loader中batch_size改为4
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))


net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
#损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()
#优化器 这里用Adam
optimizer = optim.Adam(net.parameters(), lr=0.0002)
#训练参数保存路径
save_path = './AlexNet.pth'
#训练过程中最高准确率
best_acc = 0.0

epoch_num = 10
#开始进行训练和测试，训练一轮，测试一轮
#Wirte train sequence into file for verification
#sequence_verify_file = open('seq_ver_file.txt', 'w') 
for epoch in range(epoch_num):
    # train
    net.train()    #训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0): # Train the model by the sequence of train_loader
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()    #测试过程中不需要dropout，使用所有的神经元
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')