#spile_data.py

import os
from shutil import copy, rmtree
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'flower_data/flower_photos'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]

if os.path.exists('flower_data/train'):
    rmtree('flower_data/train')
mkfile('flower_data/train')
for cla in flower_class:
    mkfile('flower_data/train/'+cla)

if os.path.exists('flower_data/val'):
    rmtree('flower_data/val')
mkfile('flower_data/val')
for cla in flower_class:
    mkfile('flower_data/val/'+cla)

if os.path.exists('flower_data/test'):
    rmtree('flower_data/test')
mkfile('flower_data/test')
for cla in flower_class:
    mkfile('flower_data/test/'+cla)

split_rate = 0.12
test_rate = 0.25
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            if random.random() <= test_rate:    #prediction test dataset
                image_path = cla_path + image
                new_path = 'flower_data/test/' + cla 
                copy(image_path, new_path)
            else: #validation dateset
                image_path = cla_path + image
                new_path = 'flower_data/val/' + cla
                copy(image_path, new_path)
        else:   #training dataset
            image_path = cla_path + image
            new_path = 'flower_data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}]Spiling processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

print("Spiling processing done!")