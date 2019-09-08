import numpy as np
import torch
import torch.utils.data as Data
from dataset import BSDS500
from U_resnet_bottleneck import Unet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = True if torch.cuda.is_available() else False

def train(batch_size=64, train_epoch=10, learning_rate=1e-4, load_model=None):
    dataset = BSDS500('./data/MSRA_images/', './data/MSRA_labels/')
    unet = Unet(3,1)
    if cuda:
        unet = unet.cuda()
    if load_model:
        unet.load_state_dict(torch.load(load_model))

    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    if cuda:
        loss_func = loss_func.cuda()

    data_loader = Data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    for epoch in range(train_epoch):
        for t, (batch_image, batch_label) in enumerate(data_loader):
            if cuda:
                batch_image = batch_image.cuda()
                batch_label = batch_label.cuda()
            prediction = unet(batch_image)

            loss = loss_func(prediction, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                'Epoch: ', epoch+1,
                '|Iter: ', t,
                '|Loss: ', loss.cpu().data.numpy()
            )
        if (epoch + 1) % 5 == 0:
            torch.save(unet.state_dict(), './models/train/train_bottleneck_' + str(epoch+1) +'.pkl')

if __name__ == '__main__':
    train(1, 100, 1e-3)
