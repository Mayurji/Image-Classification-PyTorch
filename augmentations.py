import torchvision.transforms as transforms

def augmentation(image_resolution):
    transform = transforms.Compose([transforms.ToTensor(), 
                transforms.Resize((256, 256)),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.RandomCrop(size=(image_resolution, image_resolution)),
                transforms.RandomHorizontalFlip(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform

def ContrastiveAugmentation(image_resolution):
    color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
    # 10% of the image usually, but be careful with small image sizes
    blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(size=(image_resolution, image_resolution)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([color_jitter], p=0.8),
    #transforms.RandomApply([blur], p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    return transform

