import torch
from torch.utils.data import Dataset
import os
import glob
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET

class LicensePlateDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, transforms=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transforms = transforms

        # self.__input_list = []
        self.__ann_list = []
        self.__dataAlignment()
        
    def __dataAlignment(self):
        # get_img_folders = glob.glob(self.image_paths + "/*.jpg") + glob.glob(self.image_paths + "/*.png")
        get_ann_folders = glob.glob(self.annotation_paths + "/*.xml")
        
        for ann in get_ann_folders:
            self.__ann_list.append(ann)
    

    def __getitem__(self, idx):
        # img_path = self.__input_list[idx]
        ann_path = self.__ann_list[idx]
        
        # Parse XML annotation
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        labels = []

        img_path = os.path.join(self.image_paths, root.find("filename").text)

        # Load image
        img = Image.open(img_path).convert("RGB")
        
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms:
            img = self.transforms(img)
        
        
        
        return img, target
    
    def __len__(self):
        return len(self.__ann_list)