import torch
import cv2
import numpy as np
from torchvision import transforms
import PIL
import random

from transform import RandomRotationFormSequence, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, ToTensor


class DataSample:

    def __init__(self, img, sr_factors, crop_size):
        self.img = img
        self.sr_factor = sr_factors
        self.pairs = self.create_hr_lr_pairs()
        """what's this parameter mean"""
        sizes = np.float32([x[0].size[0] * x[0].size[1] / float(img.size[0] * img.size[1]) \
                            for x in self.pairs])
        self.pair_probabilities = sizes / np.sum(sizes)

        self.transform = transforms.Compose([
            # """at last add in """,
            RandomRotationFormSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()
        ])

    def create_hr_lr_pairs(self):
        smaller_side = min(self.img.size[0:2])
        larger_side = max(self.img.size[0:2])
        # interpolation factor
        factors = []
        for i in range(smaller_side // 5, smaller_side):
            downsampled_smaller_side = i
            zoom = float(downsampled_smaller_side) / smaller_side
            dwwnsampled_larger_side = round(larger_side * zoom)
            if downsampled_smaller_side % self.sr_factor == 0 and \
                    dwwnsampled_larger_side % self.sr_factor == 0:
                factors.append(zoom)

        pairs = []
        for zoom in factors:
            hr = self.img.resize((int(self.img.size[0] * zoom), int(self.img.size[1] * zoom)),
                                 resample=PIL.Image.BICUBIC)
            lr = self.img.resize((int(hr.size[0] / self.sr_factor), int(hr.size[1] / self.sr_factor)),
                                 resample=PIL.Image.BICUBIC)

            lr = lr.resize(hr.size, resample=PIL.Image.BICUBIC)

            pairs.append((hr, lr))
            return pairs

    def generate_img(self):
        while True:
            hr, lr = random.choices(self.pairs, weights=self.pair_probabilities, k=1)[0]
            hr_tensor, lr_tensor = self.transform((hr,lr))
            hr_tensor = torch.unsqueeze(hr_tensor, 0)
            lr_tensor = torch.unsqueeze(lr_tensor, 0)
            yield hr_tensor, lr_tensor


if __name__ == '__main__':
    # cv2.namedWindow("img", 1)
    img = cv2.imread("D:\\code\\python\\getdata\\vanisa.jpg")
    print(img)
    img = PIL.Image.open("D:\\code\\python\\getdata\\vanisa.jpg")
    # print(img.shape)
    data = DataSample(img, 2, 200)
    for i, (hr,lr) in enumerate(data.generate_img()):
        print(hr)
        a = hr.numpy()
        a = a*255
        a = a.astype(np.uint8)
        print(a)
        cv2.imshow("a",a)
        cv2.waitKey(0)
        # hr, lr = data.generate_img()
        # lr.imshow("hr", hr)
        # cv2.imshow("lr", lr)

        # cv2.imshow("img", img)
        # cv2.waitKey(0)
