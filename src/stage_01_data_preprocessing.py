import torch
from torch import nn
from trochvision import datasets, transform
from torch.utils.data import Dataloader


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transform:transform.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS
):

    """
    Args:



    Returns:

    """
    train_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    test_transform = transform.Compose([
        transform.Resize(size=(64,64)),
        transform.ToTensor()
    ])



    train_data = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )



    test_data = datasets.ImageFolder(
        test_dir,
        transform= test_transform
    )

      # Get class names
    class_names = train_data.classes


    TrainDataloader = Dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory=True

    )



    TestDataloader = Dataloader(
        test_data,
        batch_size=batch_size,
        num_workers = num_workers,
        pin_memory=True

    )

    return TrainDataloader,TestDataloader, class_names
