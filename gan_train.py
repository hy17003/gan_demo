from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.init as init
import torch.optim


dataset = CIFAR10(root='./data', download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
for batch_idx, data in enumerate(dataloader):
    real_images, _ = data
    batch_size = real_images.size(0)
    print('#{} has {} images.'.format(batch_idx, batch_size))
    if batch_idx % 100 == 0:
        path = './data/CIFAR10_shuffled_batch{:03d}.png'.format(batch_idx)
        save_image(real_images, path, normalize=True)


latent_size = 64
n_channel = 3
n_g_feature = 64
gnet = nn.Sequential(
    #61*1*1
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
    nn.BatchNorm2d(4 * n_g_feature),
    nn.ReLU(),
    #256*4*4
    nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4,stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_g_feature),
    nn.ReLU(),
    #128*8*8
    nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4,stride=2, padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),
    #64*16*16
    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4,stride=2, padding=1),
    nn.Sigmoid(),
    #3*32*32
)
print(gnet)

n_d_feature = 64
dnet = nn.Sequential(
    #3*32*32
    nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2),
    #64*16*16
    nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_d_feature),
    nn.LeakyReLU(0.2),
    #128*8*8
    nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * n_d_feature),
    nn.LeakyReLU(0.2),
    #256*4*4
    nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
    #1*1*1
)
print(dnet)

def weight_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

gnet.apply(weight_init)
dnet.apply(weight_init)


goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
batch_size = 64

epoch_num = 10
criterion = torch.nn.BCEWithLogitsLoss()
fixed_noises = torch.randn(batch_size, batch_size, 1, 1)
for epoch in range(epoch_num):
    for batch_idx, data in enumerate(dataloader):
        real_images, _ = data
        batch_size = real_images.size(0)

        labels = torch.ones(batch_size)
        preds = dnet(real_images)
        outputs = preds.reshape(-1)
        dloss_real = criterion(outputs, labels)
        dmean_real = outputs.sigmoid().mean()

        noises = torch.randn(batch_size, latent_size, 1, 1)
        fake_image = gnet(noises)
        labels = torch.zeros(batch_size)
        fake = fake_image.detach()

        preds = dnet(fake)
        outputs = preds.view(-1)
        dloss_fake = criterion(outputs, labels)
        dmean_fake = outputs.sigmoid().mean()

        dloss = dloss_real + dloss_fake
        dnet.zero_grad()
        dloss.backward()
        doptimizer.step()

        labels = torch.ones(batch_size)
        preds = dnet(fake_image)
        outputs = preds.view(-1)
        gloss = criterion(outputs, labels)
        gmean_fake = outputs.sigmoid().mean()
        gnet.zero_grad()
        gloss.backward()
        goptimizer.step()

        print('[{}/{}]'.format(epoch, epoch_num) +
              '[{}/{}]'.format(batch_idx, len(dataloader))+
              '鉴别网络损失:{:g} 生成网络损失:{:g}'.format(dloss, gloss) +
              '真数据判真比例:{:g} 假数据判真比例:{:g}/{:g}'.format(dmean_real, dmean_fake, gmean_fake))
        if batch_idx % 100 == 0:
            fake = gnet(fixed_noises)
            save_image(fake, './data/images_epoch{:02d}_batch{:03d}.png'.format(epoch, batch_idx))
torch.save(gnet, 'gnet.pkl')
torch.save(dnet, 'dnet.pkl')
torch.save(gnet.state_dict(), 'gnet_params.pkl')
torch.save(dnet.state_dict(), 'dnet_params.pkl')
