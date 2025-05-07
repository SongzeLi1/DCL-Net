import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms

# normMean = [0.5, 0.5, 0.5]
# normStd = [0.5, 0.5, 0.5] # 数据归一化之前是[0,1], 归一化后是[-1,1], (0-mean)/std=(0-0.5)/0.5=-1, (1-mean)/std=(1-0.5)/0.5=1
# # 'coco'
# normMean = [0.471, 0.448, 0.408]
# normStd = [0.234, 0.239, 0.242]
# # 'imagenet'
# normMean = [0.485, 0.456, 0.406]
# normStd = [0.229, 0.224, 0.225]
# Ali_new
# normMean = [0.85811307, 0.86725448, 0.87162606]
# normStd = [0.15834805, 0.15616383, 0.15007581]
# docimg


# train_transform = [
#     transforms.Resize(224),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
#     ], p=0.5),
#     transforms.RandomGrayscale(p=0.5),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=(-90, 90)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ]


# No augmentation
def train_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size[0] == input_size[1]:
        return A.Compose([
        A.Resize(input_size[0], input_size[0]),  # width, height
        # A.ImageCompression(quality_lower=70, quality_upper=100, p=1), # 测png的时候不加压缩去训练
        A.Normalize(mean=normMean, std=normStd),
        ToTensorV2(),
    ])


# augmentation
# def train_transform(input_size):
#     normMean = [0.485, 0.456, 0.406]
#     normStd = [0.229, 0.224, 0.225]
#     return A.Compose([
#     A.Resize(input_size[0], input_size[0]), # width, height
#     A.OneOf([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.HueSaturationValue(p=0.5),
#     ]),
#     A.MedianBlur(p=0.3),
#     A.GaussNoise(var_limit=(25.0, 900.0), mean=0, p=0.3),
#     A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
#     A.Normalize(mean=normMean, std=normStd),
#     ToTensorV2(),
# ])

# # augmentation
# def train_transform(input_size):
#     normMean = [0.485, 0.456, 0.406]
#     normStd = [0.229, 0.224, 0.225]
#     if input_size[0] == input_size[1]:
#         return A.Compose([
#         A.Resize(input_size[0], input_size[0]), # width, height
#         A.RandomResizedCrop(input_size[0], input_size[0], scale=(0.8, 1.0)),
#         A.HorizontalFlip(p=0.5),
#         A.OneOf([
#             A.VerticalFlip(p=0.5),
#             # A.RandomRotate90(p=0.5),
#             # A.RandomBrightnessContrast(p=0.5),
#             # A.HueSaturationValue(p=0.5),
#             # A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#             # # A.CoarseDropout(p=0.2),
#             # A.Transpose(p=0.5)
#         ]),
#         # A.ImageCompression(quality_lower=70, quality_upper=100, p=1), # 测png的时候不加压缩去训练
#         A.Normalize(mean=normMean, std=normStd),
#         ToTensorV2(),
#     ])
#     else:
#         return A.Compose([
#             A.Resize(input_size[0], input_size[1]),
#             A.HorizontalFlip(p=0.5),
#             A.OneOf([
#                 A.VerticalFlip(p=0.5),
#                 A.RandomBrightnessContrast(p=0.5),
#                 A.HueSaturationValue(p=0.5),
#                 A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#                 # A.CoarseDropout(p=0.2),
#             ]),
#             A.ImageCompression(quality_lower=70, quality_upper=100, p=1), # 测png的时候不加压缩去训练
#             A.Normalize(mean=normMean, std=normStd),
#             ToTensorV2(),
#         ])


def val_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    return A.Compose([
    A.Resize(input_size[0], input_size[1]),
    A.Normalize(mean = normMean, std = normStd),
    ToTensorV2(),
])


def infer_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size is not None:
        return A.Compose([
            A.Resize(input_size[0], input_size[1]), # width, height
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])