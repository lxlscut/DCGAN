import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.nn import init
import tqdm
from torchvision.models.detection import transform
from torchvision.transforms import transforms

from net import ZSSRNet
from getdata import DataSample
import torch
import torch.nn as nn
import PIL

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


# change learning_rate automatic
def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(model, img, sr_factor, num_batches, learning_rate, crop_size):
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    sampler = DataSample(img,sr_factor,crop_size)
    with tqdm.tqdm(total = num_batches,miniters = 1,mininterval=0) as progress:
        for iter,(hr,lr) in enumerate(sampler.generate_img()):

            model.zero_grad()

            lr = lr.to(device)
            hr = hr.to(device)

            # lr = Variable(lr).cuda()
            # hr = Variable(hr).cuda()

            output = model(lr) + lr

            error = loss(output, hr)

            # cpu_loss = error.data.cpu().numpy()
            progress.set_description("Iteration: {iter} Loss : {loss},Leaning Rate:{lr}".format(\
                iter=iter, loss=error, lr=learning_rate))
            progress.update()

            if iter > 0 and iter % 10000 == 0:
                learning_rate = learning_rate / 10
                adjust_learning_rate(optimizer, new_lr=learning_rate)
                print("Learning rate reduced to {lr}".format(lr = learning_rate))

            error.backward()
            optimizer.step()

            if iter > num_batches:
                print("Done Training")
                break

def test(model,img,sr_factor):
    with torch.no_grad():
    # model.to(device)
        model.eval()
        img = img.resize((int(img.size[0]*sr_factor),\
                          int(img.size[1]*sr_factor)),resample = PIL.Image.BICUBIC)
        img.save('low_res.png')
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img,0)
        input = Variable(img.cuda())
        # input = img.to(device)
        residual = model(input)
        output = input + residual
        output = output.cuda().data[0, :, :, :]
        o = output.numpy()
        o[np.where(o<0)] = 0.0
        o[np.where(0>1)] = 1.0
        output = torch.from_numpy(o)
        output = transform.ToPILImage()(output)
        output.save('zssr.png')

if __name__ == '__main__':
    img = PIL.Image.open("D:\\code\\python\\getdata\\vanisa.jpg")
    num_channels = len(np.array(img).shape)
    if num_channels == 3:
        model = ZSSRNet(input_channels = 3).to(device)
    else:
        print("Expecting RGB or gray image,instead got", img.size)
    model.apply(weights_init_kaiming)

    train(model, img, 2, 15000, 0.00001, 128)
    test(model,img,2)