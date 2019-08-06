import torch
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np

dnet = torch.load('dnet.pkl')
gnet = torch.load('gnet.pkl')

def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

batch_size = 64

while(1):
    noises = torch.randn(batch_size, batch_size, 1, 1)
    fake = gnet(noises)
    img = torchvision.utils.make_grid(fake)
    imshow(img)
    plt.show()
