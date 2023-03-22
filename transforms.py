import torch
import numpy as np
from torchvision import transforms


class Preprocessing(object):
    def __init__(self,
                 means=(0.485, 0.456, 0.406),
                 stds=(0.229, 0.224, 0.225),
                 augmentor=None):
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(means, stds)])
        self.augmentor = augmentor

    def __call__(self, sample: dict):
        if self.augmentor is not None:
            sample = self.augmentor(**sample)
            del sample['transform_params']

        res = list(sample.values())
        labels = []
        img = self.trans(res[0])

        for i in range(1, len(res)):
            res[i] = torch.tensor(res[i])
            labels.append(torch.full((len(res[i]),), i - 1))
            res[i] = res[i].reshape(-1)

        return img, torch.cat(res[1:]), torch.cat(labels)
