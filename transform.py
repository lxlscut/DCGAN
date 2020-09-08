import numpy as np
import random
from torchvision.transforms import functional as F
import numbers


class RandomRotationFormSequence(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degree):
        # degree is a array,chose one element on the array
        angle = np.random.choice(degree)
        return angle

    def __call__(self, data):
        hr, lr = data
        angle = self.get_params(self.degrees)
        return F.rotate(hr, angle, self.resample, self.expand, self.center), \
               F.rotate(lr, angle, self.resample, self.expand, self.center)


# random  horizontal  flip function
class RandomHorizontalFlip(object):
    def __call__(self, data):
        hr, lr = data
        if random.random() < 0.5:
            return F.hflip(hr), F.hflip(lr)
        return hr, lr


# random vertical flip function
class RandomVerticalFlip(object):
    def __call__(self, data):
        hr, lr = data
        if random.random() < 0.5:
            return F.vflip(hr), F.vflip(lr)
        return hr, lr


# random crop the image
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_param(data, output_size):
        hr, lr = data
        w, h = hr.size
        th, tw = output_size
        if w == tw or h == th:
            return 0, 0, h, w
        if w < tw or h < th:
            th, tw = h//2, w//2

        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)

        return i, j, th, tw

    def __call__(self, data):
        hr, lr = data
        if self.padding > 0:
            hr = F.pad(hr, self.padding)
            lr = F.pad(lr, self.padding)
        i, j, h, w = self.get_param(data, self.size)
        return F.crop(hr, i, j, h, w), F.crop(lr, i, j, h, w)


class ToTensor(object):
    def __call__(self, data):
        hr, lr = data
        return F.to_tensor(hr),F.to_tensor(lr)







