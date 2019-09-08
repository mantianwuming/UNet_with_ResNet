import torch.utils.data as Data
import torch
import os
import cv2
import numpy as np

class BSDS500(Data.Dataset):
    def __init__(self, image_path, label_path):
        self.image_name = np.arange(len(os.listdir(image_path)))
        self.image_path = image_path
        self.label_path = label_path

    def __getitem__(self, index):
        image, label = self.pull_item(index)
        return image, label

    def __len__(self):
        return len(self.image_name)

    def pull_item(self, index):
        image_path = self.image_path
        image_index = self.image_name[index]
        image_name = str(image_index) + '.jpg'
        image_name = os.path.join(image_path, image_name)
        image = cv2.imread(image_name)
        image = image / 255.0
        image = image.transpose([2,0,1])
        image = torch.from_numpy(image).type(torch.FloatTensor)

        label_path = self.label_path
        label_name = str(image_index) + '.bmp'
        label_name = os.path.join(label_path, label_name)
        label = cv2.imread(label_name, 0)
        label = label / 255.0
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return image, label

if __name__ == "__main__":
    dataset = BSDS500('./data/MSRA_images', './data/MSRA_labels')
    print(len(dataset))
    data_loader = Data.DataLoader(dataset, 16, num_workers=0, shuffle=True, pin_memory=True)
    for t, (batch_image, batch_label) in enumerate(data_loader):
        print(batch_image.size(), batch_label.size())
