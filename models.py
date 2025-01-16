from object_detection.faster_rcnn import PretrainedFasterRCNN
import torch.nn as nn
import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torchvision.transforms.functional import crop, resize
from transformers import ViTImageProcessor, ViTModel
from utils import load_yaml

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfAttention(nn.Module): 
    def __init__(self, glove_dim=300): 
        super().__init__() 
        self.glove_dim = glove_dim  
        self.W_1 = nn.Linear(self.glove_dim, 1).to(device)
        self.W_2 = nn.Linear(self.glove_dim, self.glove_dim).to(device)
        self.W_3 = nn.Linear(self.glove_dim, 1).to(device)
        self.tanh = nn.Tanh() 

    def _get_weights(self, values):   
        z = self.W_1(values) 
        weights = nn.functional.softmax(z, dim=1) 
        return weights
    
    def forward(self, values):  
        weights = self._get_weights(values) 
        new_ftr = torch.mul(weights, values) 
        new_ftr = new_ftr.sum(1) 
        return new_ftr, weights 
    
class ObjectAttention(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.W_1 = nn.Linear(feature_dim, feature_dim).to(device) 
        self.W_2 = nn.Linear(feature_dim, 1).to(device) 
        
    def _get_weights(self, values):
        """
        Args:
            values: shape [batch_size, seq_len, feature_dim]
        Returns:
            weights: shape [batch_size, seq_len, 1]
        """
        transformed = self.W_1(values)  # [batch_size, seq_len, feature_dim]
        scores = self.W_2(transformed)  # [batch_size, seq_len, 1]
        weights = nn.functional.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        return weights

    def forward(self, values):
        """
        Args:
            values: shape [batch_size, seq_len, feature_dim]
        Returns:
            new_ftr: shape [batch_size, feature_dim]
            weights: shape [batch_size, seq_len, 1]
        """
        values = values.to(device)
        batch_size, seq_len, feature_dim = values.shape
        weights = self._get_weights(values)  # [batch_size, seq_len, 1]
        weighted_values = values * weights  # [batch_size, seq_len, feature_dim]
        new_ftr = weighted_values.sum(dim=1)  # [batch_size, feature_dim]
        return new_ftr, weights

class LocalBranch:   
    def __init__(self, k=10, glove_path='./EmojiGenerator/example/emoji-gan/utils/glove.6B.300d.txt'):
        self.k = k 
        self.glove_dim = 300 
        self.glove = self.load_glove(glove_path, self.glove_dim) 
        self.fasterrcnn = PretrainedFasterRCNN()

    def load_glove(self, data_dir_path=None, embedding_dim=None):
        if embedding_dim is None:
            embedding_dim = 300

        glove_file_path = './EmojiGenerator/example/emoji-gan/utils/glove.6B.300d.txt'
        _word2em = {}
        file = open(glove_file_path, mode='rt', encoding='utf8')
        for line in tqdm(file):
            words = line.strip().split()
            word = words[0]
            embeds = np.array(words[1:], dtype=np.float32)
            _word2em[word] = embeds
        file.close()
        return _word2em
        
    def _denormalize(self, img):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return np.array(img.cpu().permute(1, 2, 0) * 255).astype(np.uint8)
        
    def get_object_embeddings(self, img): 
        word_emb_tensor = []
        for i in img:
            classes = self.fasterrcnn.detect_object(self._denormalize(i)) 
            word_embs = torch.stack([torch.FloatTensor(self.glove[j]) if j in self.glove else torch.zeros(self.glove_dim) for j in classes])
            word_emb_tensor.append(word_embs) 
        return torch.stack(word_emb_tensor)

class ObjectBranch:
    def __init__(self):
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', from_tf=True)
        self.vit = self.vit.to(device)
        self.fasterrcnn = PretrainedFasterRCNN()

    def _denormalize(self, img):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return np.array(img.cpu().permute(1, 2, 0) * 255).astype(np.uint8)
    
    def get_object_features(self, img):
        PROJECT_DIR = os.path.dirname(__file__)
        train_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'train_config.yaml'))
        BATCH_SIZE = train_config['DATALOADER']['batch_size']
        object_features = []
        target_length = len(img)

        batched_cropped_images = []
        batched_image_indices = []

        for idx, i in enumerate(img):
            boxes = self.fasterrcnn.detect_boxes(self._denormalize(i))
            boxes_tensor = torch.as_tensor(boxes).clone().detach().view(-1, 4)
            sorted_indices = torch.argsort((boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]), descending=True)
            sorted_boxes = boxes_tensor[sorted_indices]

            for box_idx, box in enumerate(sorted_boxes[:10]): 
                top = int(box[1])
                left = int(box[0])
                height = int(box[3] - box[1])
                width = int(box[2] - box[0])

                if height == 0 or width == 0:
                    continue

                cropped_img = i[:, top:top + height, left:left + width]

                if cropped_img.numel() == 0:
                    continue

                resized_img = F.interpolate(cropped_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                batched_cropped_images.append(resized_img)
                batched_image_indices.append(idx)

        if batched_cropped_images:
            batched_cropped_images = torch.cat(batched_cropped_images, dim=0)

            inputs = self.image_processor(images=batched_cropped_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = self.vit(**inputs)

            for batch_idx, img_idx in enumerate(batched_image_indices):
                feature = outputs.last_hidden_state[batch_idx].unsqueeze(0)
                object_features.append(feature)

        if object_features:
            object_features_tensor = torch.cat(object_features, dim=0) 
            mean_tensor = object_features_tensor.mean(dim=0, keepdim=True)
            object_features_tensor = mean_tensor.repeat(target_length, 1, 1) 
            
            return object_features_tensor
        
class OSANet(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(OSANet, self).__init__()
        self.num_classes = num_classes
        feature_dim = 768
        glove_dim = 300
        
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', from_tf=True)
        self.vit = self.vit.to(device)

        self.att_list = nn.ModuleList([SelfAttention() for _ in range(self.num_classes)])
        self.obj_att_list = nn.ModuleList([ObjectAttention(feature_dim) for _ in range(self.num_classes)])
        self.lin_list = nn.ModuleList([nn.Linear(glove_dim, 1) for _ in range(self.num_classes)])
        
        self.k = 10
        self.dropout = nn.Dropout(p=0.5)
        self.linear_1 = nn.Linear(feature_dim, glove_dim).to(device)
        self.linear_2 = nn.Linear(glove_dim, 1).to(device)
        self.linear_3 = nn.Linear(glove_dim + feature_dim, self.num_classes).to(device)

    def forward(self, img, obj, obj_add):
        # img: (batch, 3, 224, 224)
        # obj: (batch, 10, 300)
        # obj_add: (batch, 197, 768)
        
        img = img.to(device)
        obj = obj.to(device)
        obj_add = obj_add.to(device)
        
        batch_size = img.size(0)
        
        # Global branch
        g = self.vit(img)['last_hidden_state']  # [batch_size, 197, 768]
        g_cls = g[:, 0, :]  # [batch_size, 768]
        g_prime = self.linear_1(g_cls)  # [batch_size, 300]
        
        weight_list = []
        b_list = []
        h = torch.zeros(batch_size, self.num_classes).to(device)

        for idx in range(self.num_classes):
            # Semantic branch
            _, weights = self.att_list[idx](obj)  # obj: [batch_size, 10, 300]
            weight_list.append(weights)
            
            # Object branch
            b, b_weights = self.obj_att_list[idx](obj_add)  # b: [batch_size, 768]
            b_list.append(b_weights)
            
            # 여기서 g_prime은 [batch_size, 300]이고 b는 [batch_size, 768]
            combined_features = torch.cat([b, g_prime], dim=1)  # [batch_size, 768+300]
            
            output = self.linear_3(combined_features)
            h += output

        h = self.dropout(h)
        return h, weight_list
