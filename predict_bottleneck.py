import numpy as np
import torch
import os
import cv2
from dataset import BSDS500
from U_resnet_bottleneck import Unet
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = True if torch.cuda.is_available() else False
dataset = BSDS500('./data/MSRA_images', './data/MSRA_labels')

def get_unet(load_model=None):
    unet = Unet(3,1)
    if cuda:
        unet = unet.cuda()
    if load_model:
        print('Loading parameters of model.')
        unet.load_state_dict(torch.load(load_model))
    unet.eval()
    return unet

def predict(unet, number = [], image_path='./data/test/', image=None):
    if image:
        img = cv2.imread(image)
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)
            image_name = os.path.join(image_path, image)
            cv2.imwrite(image_name, img)
        img = cv2.resize(img, (400, 320), interpolation=cv2.INTER_CUBIC)
        image_transform = img.astype(np.float32) / 255.0
        image_transform = image_transform.transpose([2,0,1])
        image_transform = np.expand_dims(image_transform, axis=0)
        image_transform = torch.from_numpy(image_transform).type(torch.FloatTensor)
        if cuda:
            image_transform = image_transform.cuda()
        prediction = unet(image_transform)
        temp = prediction.cpu().data.numpy()
        tmax = np.amax(temp)
        tmin = np.amin(temp)
        print(tmax)
        print(tmin)
        temp = temp[0,0]

        test_number = tmax-(tmax-tmin)*0.3
        print(test_number)
        b = np.ones(temp.shape)*test_number
        temp = temp > b
        temp = temp.astype(np.uint8)*255
        plt.figure()
        plt.subplot(121)
        plt.imshow(img[:,:,::-1])
        plt.subplot(122)
        plt.imshwo(temp)
        plt.show()
    else:
        img_name = os.listdir(image_path)
        for i in range(len(number)):
            image_name = os.path.join(image_path,img_name[number[i]])
            img = cv2.imread(image_name)
            if img.shape[0] > img.shape[1]:
                img = np.rot90(img)
                cv2.imwrite(image_path + image_name[number[i]], img)
            img = cv2.resize(img, (400, 320), interpolation = cv2.INTER_CUBIC)
            image_transform = img.astype(np.float32) / 255.0
            image_transform = image_transform.transpose([2,0,1])
            image_transform = np.expand_dims(image_transform, axis=0)
            image_transform = torch.from_numpy(image_transform).type(torch.FloatTensor)
            if cuda:
                image_transform = image_transform.cuda()
            prediction = unet(image_transform)
            temp = prediction.cpu().data.numpy()
            temp = prediction.cpu().data.numpy()
            tmax = np.amax(temp)
            tmin = np.amin(temp)
            print(tmax)
            print(tmin)
            temp = temp[0]

            test_number = tmax-(tmax-tmin)*0.8
            print(test_number)
            b = np.ones(temp.shape)*test_number
            temp = temp>b
            temp = temp.astype(np.uint8)*255
            plt.figure()
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(temp)
            plt.show()

if __name__ == '__main__':
    unet = get_unet('./models/train/train_bottleneck_100.pkl')
    number = list(range(100))
    predict(unet, number, image_path='./data/MSRA_images/', image=None)
