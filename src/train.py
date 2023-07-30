import os
import torch
# import TinyVGG, create_dataloaders,train
from utils import utils
from datetime import datetime
from data_preprocessing import create_dataloaders
from model_creation import TinyVGG
from training_steps import train
import argparse
import torchvision






def main(config_path,target_dir,model_name):
    configs = utils.read_yaml(config_path)
    train_dir = os.path.join(configs["data"]["data_folder"],os.path.join(configs["data"]["image_folder"],"train"))
    test_dir = os.path.join(configs["data"]["data_folder"],os.path.join(configs["data"]["image_folder"],"test"))
    batch_size = configs["config"]["BATCH_SIZE"]
    learning_rate = configs["config"]["learning_rate"]
    epochs = configs["config"]["epochs"]
    device = configs["config"]["device"]
    # num_workers = configs["config"]["NUM_WORKERS"]

    train_loader,test_loader, classes = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size,
    #    transform = defined in the internal code only we can redefined outside here if neeeded
        # num_workers=num_workers
    )
    # print("*"*18)
    # print(classes)
    # print("*"*18)
    # img_batch, label_batch = next(iter(train_loader))
    # print(img_batch)

    model = TinyVGG(
        input_shape = configs["config"]["in_channels"],
        hidden_units = configs["config"]["hidden_units"],
        output_shape = len(classes),

    ).to(device)


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate
        )

    train(
        model= model,
        train_dataloader= train_loader,
        test_dataloader = test_loader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs=epochs,
        device = device
    )

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir = target_dir,
                    model_name = str(model_name) +str(".pth"))


if __name__=="__main__":
    dt = datetime.now()
    uniqueness = str(dt).replace(" ","_").replace(":","__").split(".")[0]
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="configs.yaml")
    args.add_argument("--target_dir","-t",default="models")
    args.add_argument("--model_name","-m",default="Tiny_VGG_model"+str(uniqueness))
    parsed_args = args.parse_args()
    try:
        main(parsed_args.config,parsed_args.target_dir,parsed_args.model_name)
    except Exception as e:
        raise e
    
