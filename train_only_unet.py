import numpy as np
import torch
import torch.utils.data as Data
from dataset import BSDS500
from UNET_1 import Unet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda = True if torch.cuda.is_available() else False

def train(model_name='resnet', batch_size=64, train_epoch=10, learning_rate=1e-4, load_model=None):
    """
    Train.
    :param model_name: pre-train model. a str
    :param batch_size: batch size. a int
    :param train_epoch: How many epoch. a int
    :param learning_rate: learning rate. a float
    :param load_model: the parameters of classifier model which has been trained. a str(folder)
    """
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

    # def l1_penalty(var):
    #     return torch.abs(var).sum()

    # def l2_penalty(var):
    #     return torch.sqrt(torch.pow(var, 2).sum())


    data_loader = Data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    for epoch in range(train_epoch):
        for t, (batch_image, batch_label) in enumerate(data_loader):
            if cuda:
                batch_image = batch_image.cuda()
                batch_label = batch_label.cuda()
            prediction = unet(batch_image)


            # lambda1, lambda2 = 0.5, 0.01
            # l1_regularization = lambda1 * l1_penalty(prediction)
            # l2_regularization = lambda2 * l2_penalty(prediction)

            # loss = loss_func(prediction, batch_label) + l1_regularization + l2_regularization
            loss = loss_func(prediction, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                'Epoch', epoch+1,
                '|Iter', t,
                '|Loss:', loss.cpu().data.numpy()
            )
        if (epoch+1) % 5 == 0:
            torch.save(unet.state_dict(), './models/train/train_only_unet_' + str(epoch+1) + '.pkl')

if __name__ == '__main__':
    train(None, 1, 100, 1e-3)
