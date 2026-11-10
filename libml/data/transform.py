# -*- coding: utf-8 -*-


import torchvision.transforms as T


IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

def build_transform_default(is_train=True, input_size=224):
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform_list = [
             T.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
             T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ]
        return transform_list

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            T.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(T.CenterCrop(input_size))
    t.append(T.ToTensor())
    
    return t 
# def build_imagenet_transform(
#     train: bool = True,
#     interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
#     norm: bool = False,
# ):
#     transforms = [T.ToTensor()]
#     if norm:
#         transforms.append(T.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD))

#     if train:
#         train_transforms = [
#             T.RandomResizedCrop(224, interpolation=interpolation),
#             T.RandomHorizontalFlip(),
#         ]
#         return train_transforms + transforms

#     test_transforms = [
#         T.Resize(256, interpolation=interpolation),
#         T.CenterCrop(224),
#     ]
#     return test_transforms + transforms
