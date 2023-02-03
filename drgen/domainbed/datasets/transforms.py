from torchvision import transforms as T


basic = T.Compose(
    [
        # T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
aug = T.Compose([
        T.RandomResizedCrop(224,scale=(0.8, 1.2),ratio=(0.8,1.2)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.RandomAffine(degrees=(-180,180),
                translate=(0.2,0.2)),
        T.GaussianBlur(kernel_size=7, sigma=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



# #aug_args.translation.range,
#                 fillcolor=aug_args.value_fill
#             ))
#     ]
# )
