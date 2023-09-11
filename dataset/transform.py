import requests
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import albumentations as A
from core import config

D_trans = A.Compose([
    A.Resize(256,256),
    # A.Flip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # # A.ShiftScaleRotate(p=0.3),
    # A.VerticalFlip(p=0.5),
    # A.HorizontalFlip(p=0.5),
    # A.Transpose(p=0.5),
])
B_trans = A.Compose([
    A.Resize(256,256),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate(p=0.3),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.Transpose(p=0.5),
])
A_trans = A.Compose([])
C_trans = A.Compose([
    A.OneOf([
        A.CLAHE(p=1),
        # A.RandomBrightness(p=0.2),
        A.RandomBrightnessContrast(p=1),
        # A.RandomContrast(p=0.2),
        A.RandomGamma(p=1),
    ],p=0.15),
    # ToTensor(),
    # Lambda(lambda t: t / 255.),
    # Lambda(lambda t: (t * 2) - 1),
])
transform = Compose([
    # Resize(image_size),
    # CenterCrop(image_size),
    ToTensor(),
    # Lambda(lambda t: t / 255.),# turn into Numpy array of shape HWC, divide by 255
    Lambda(lambda t: (t * 2) - 1),
])


reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])


