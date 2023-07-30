import requests
import zipfile
import torch
import yaml
from torch import nn
from pathlib import Path
from trochvision import datasets, transform
from torch.utils.data import Dataloader
from src.utils import read_yaml


def data_extraction():
    configs = read_yaml("./configs.yaml")
    # Setup path to data folder
    data_path = Path(configs["data"]["data_folder"])
    image_path = data_path / configs["data"]["image_folder"]

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get(configs["data"]["data_source"])
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...") 
            zip_ref.extractall(image_path)



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










