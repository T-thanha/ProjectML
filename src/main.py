#<---------------------------------> Outsource Libraries <-------------------------->
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torchvision.transforms as transforms
#<---------------------------------> Internal Import <-------------------------->
from learning.datasets import LicensePlateDataset
from learning.networks import FasterRCNN
from learning.models import train_model
#<---------------------------------> Define <-------------------------->
# PATH
IMAGES_PATH = ["data/images/", "data/images2/"]
LABEL_PATH = ["data/annotations/", "data/annotations2/"]
MODEL_SAVE_PATH = "models/RCNN/"
# = Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001

# - Model State
MODEL_CHECKPOINT = "MLP-best-train_loss148.6988-epoch=11-lr=0.002-wd=0.0001.pt"
RUN_EPOCHS = 50

if __name__ == "__main__":
    # - > Transform
    mtransforms = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # -> Load Dataset
    dataset_list = []
    for img_path, label_path in zip(IMAGES_PATH, LABEL_PATH):
        dataset_list.append(LicensePlateDataset(img_path, label_path, transforms=mtransforms))
    dataset = ConcatDataset(dataset_list)

    # -> Data split
    split_generator = torch.Generator().manual_seed(999)

    train_dataset, val_dataset = random_split(dataset, [0.90, 0.10], generator=split_generator)

    # -> Data Loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # -> Device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # -> Model
    model = FasterRCNN(num_classes=2)
    
    # -> Train Model
    trained_model = train_model(device, model, train_loader,val_loader=val_loader,lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,
                                num_epochs=RUN_EPOCHS, model_save_path=MODEL_SAVE_PATH,model_name=MODEL_CHECKPOINT)
