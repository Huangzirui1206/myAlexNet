#predict.py

from tkinter import image_names
from matplotlib.lines import lineStyles
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import time

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()


# Write predict result into text file
test_result = None
if not os.path.exists('./result/test_result.txt'):
    test_result = open('./result/test_result.txt', 'w')
else:
    test_result = open('./result/test_result.txt', 'a')
test_result.write('Test ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')


# Test samples in 'flower_data/test'
file = 'flower_data/test'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
result_dict = {}
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    accuracy = 0
    for index, image in enumerate(images):
        image_path = cla_path + image
        img = Image.open(image_path)
        # [N, C, H, W]
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
        # predict class
            output = torch.squeeze(model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        test_result.write(f'{cla}[{index}], {class_indict[str(predict_cla)]}, {predict[predict_cla].item()}\n')
        if cla == class_indict[str(predict_cla)]:
            accuracy += 1
    accuracy = accuracy / len(images)
    result_dict[cla] = accuracy
result_dict['average'] = sum([value for value in result_dict.values()]) / len(result_dict)
test_result.write('\n\n')
test_result.close()

#Draw result picture
x = [i for i in range(len(flower_class) + 1)]
y = list(result_dict.values())
x_labels = list(result_dict.keys())
color = ['red', 'blue']
plt.xticks(x, x_labels)
plt.bar(x, y, width=[0.5 for i in range(len(x))], color=color)
plt.grid(True, linestyle=":" , color = 'b', alpha = 0.6)
plt.show()

        


#Quick_test for one picture        
# load image
#Here fold ./quick_test is deleted
#img = Image.open("./quick_test/sunflower.jpg")  #验证太阳花 
#img = Image.open("./quick_test/rose.jpg")     #验证玫瑰花
#plt.imshow(img)
# [N, C, H, W]
#img = data_transform(img)
# expand batch dimension
#img = torch.unsqueeze(img, dim=0)

#with torch.no_grad():
    # predict class
#    output = torch.squeeze(model(img))
#    predict = torch.softmax(output, dim=0)
#    predict_cla = torch.argmax(predict).numpy()
#print(class_indict[str(predict_cla)], predict[predict_cla].item())
#plt.show()
