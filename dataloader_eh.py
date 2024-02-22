import os
import glob
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def read_dicts_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            dictionaries = []
            for line in file:
                dictionary = json.loads(line.strip())
                dictionaries.append(dictionary)
        return dictionaries
    except IOError:
        print(f"Could not open or read the file: {file_path}")


class DepthDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = sorted(glob.glob(os.path.join(root_dir, 'annotations/*')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load depth value
        loaded_dicts = read_dicts_from_file(file_path)
        depth_value = loaded_dicts[0]['gt_puck']
        keypoints = loaded_dicts[0]['keypoints']
        bbox = loaded_dicts[0]['bbox']

        image_path = file_path.replace('annotations', 'rgb').replace('.txt', '_rgb.png')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        #print('STARTING')
        #print(f'file_path: {file_path}, depth_value {depth_value} and keypoints: {keypoints}')
        
        return (file_path, image, depth_value, keypoints, bbox)
    

class FloorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.transform_RGB =   transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                                   transforms.Resize((256,256)),
                                                   transforms.ToTensor()])
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpeg", "_seg.jpeg"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")

        if self.transform:
            image = self.transform_RGB(image).float()
            mask = self.transform(mask).int()#.squeeze()



        #normalize images
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        
        

        return image, mask


class FloorADE20kDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.transform_RGB =   transforms.Compose([transforms.Lambda(lambda img: img.convert('RGB')),
                                                   transforms.Resize((256,256)),
                                                   transforms.ToTensor()])
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", "_seg.jpg"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")

        if self.transform:
            image = self.transform_RGB(image).float()
            mask = self.transform(mask).int()#.squeeze()



        #normalize images
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        
        

        return image, mask

