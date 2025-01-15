from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import os
from PIL import Image
import torch
import numpy as np
from sklearn import preprocessing

class dataset(Dataset):
    def __init__(self, data_name, le, mode, data_dir):
        data_dir_path = os.path.join(data_dir, data_name, mode)
        self.file_list = []
        self.path_list = []

        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        if os.path.isdir(data_dir_path):
            for emo in os.listdir(data_dir_path):
                emo_dir = os.path.join(data_dir_path, emo)
                if os.path.isdir(emo_dir):
                    for i in os.listdir(emo_dir):
                        img_path = os.path.join(emo_dir, i)
                        if os.path.isfile(img_path) and img_path.lower().endswith(valid_extensions):
                            self.file_list.append(i)
                            self.path_list.append(img_path)
                            
        if data_name == 'EmoSet' or data_name == 'EmoSet-2' or data_name == 'FI' or data_name == 'FI-2' or data_name == 'flickr' or data_name == 'instagram':
            self.name_list = [int(i.split('_')[-1].split('.')[0]) if i.split('_')[-1].split('.')[0].isdigit() else -1 for i in self.file_list]
            
        all_labels = ['train', 'test', 'val'] + [i.split('/')[-2] for i in self.path_list]
        le.fit(all_labels)
        self.label_list = [le.transform([i.split('/')[-2]])[0] for i in self.path_list]  
        self.mode = mode    
        self.pos_emotion = ['amusement', 'contentment', 'awe', 'excitement']
        self.neg_emotion = ['anger', 'disgust', 'fear', 'sadness']

        input_size = 224
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def __len__(self):
        return len(self.path_list) 
    
    def __getitem__(self, item):   
        image_path = self.path_list[item]
        image_name_idx = self.name_list[item] 
        image = Image.open(image_path).convert("RGB")
        # resize image
        image = self.transforms[self.mode](image) 
        
        label = self.label_list[item]              
    
        return { 
          'idx' : torch.tensor(image_name_idx, dtype=torch.int64) ,
          'image' : image,  
          'label': torch.tensor(label, dtype=torch.int64) 
        }

def get_label_encoder(dir):
    label_list = []
    abs_dir = os.path.abspath(dir)
    for i in os.listdir(abs_dir):
        label_list.append(i.split('_')[0])
    label_list = list(set(label_list))

    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    return le

def load_dataloader(data_name, le, batch_size, mode, data_dir, num_workers=0): 
    ds = dataset(data_name = data_name, le=le, mode=mode, data_dir=data_dir) 
    data_loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle = True, num_workers=num_workers)
    return data_loader
