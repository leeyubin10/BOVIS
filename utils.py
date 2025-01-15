import torch
import numpy as np
import yaml

class EmoLoss(torch.nn.Module):
    '''Calculate emotional loss using Mikel's Wheel distance'''
    def __init__(self):
        super(EmoLoss, self).__init__()
        self.idx_to_emo = {0:'amusement', 1:'anger', 2:'awe', 3:'contentment', 4:'disgust', 5:'excitement', 6:'fear', 7:'sadness'}
        self.emo_to_idx_loss = {'fear':0, 'excitement':1, 'awe':2, 'contentment':3, 'amusement':4, 'anger':5, 'disgust':6, 'sadness':7}
        #self.idx_to_emo = { 0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'sadness', 5: 'surprise' }
        #self.emo_to_idx_loss = {'fear':0, 'surprise':1, 'joy':2, 'anger':3, 'disgust':4, 'sadness':5}

    def _convert_to_new_idx(self, input_list):
        '''Convert to new index which indicates the position in Mikel's Wheel'''
        emo_list = [self.idx_to_emo[int(i.item())] for i in input_list]
        new_idx_list = [self.emo_to_idx_loss[e] for e in emo_list]
        return torch.LongTensor(new_idx_list)
    
    def _get_min_dist(self, pred, true):
        dist1 = torch.abs(pred - true)
        dist2 = torch.abs(pred + 2 - true)
        dist3 = torch.abs(true + 2 - pred)
        #dist2 = torch.abs(pred + 6 - true)
        #dist3 = torch.abs(true + 6 - pred)
        
        total_dist = torch.stack((dist1, dist2, dist3))
        min_dist = torch.min(total_dist, dim=0)[0].float()
        return min_dist
    
    def forward(self, logits, labels):
        '''
        # Arguments
         - logits : (batch_size, 1)
         - labels : (batch_size, 1)
        '''
        assert logits.size() == labels.size()
        new_pred_list = self._convert_to_new_idx(logits)
        new_true_list = self._convert_to_new_idx(labels)
        min_dist = self._get_min_dist(new_pred_list, new_true_list)
        
        return torch.mean(min_dist)

class GMAELoss(torch.nn.Module):
    def __init__(self):
        super(GMAELoss, self).__init__()

    def forward(self, preds, labels):
        '''
        Compute the GMAE loss.
        
        Args:
            preds (torch.Tensor): Predictions from the model, shape (batch_size,).
            labels (torch.Tensor): True labels, shape (batch_size,).
        
        Returns:
            torch.Tensor: GMAE loss.
        '''
        abs_diff = torch.abs(preds - labels)
        exp_abs_diff = torch.exp(abs_diff)  # e^|Y - \hat{Y}|
        loss = -2 * torch.log(exp_abs_diff + 1) + 2 * abs_diff
        return torch.mean(loss)

def compute_bias_weight(preds, labels, strategy='MinStrategy'):
    """
    Compute bias weight for each sample based on the given strategy.

    Args:
        preds (torch.Tensor): Predictions from the model, shape (batch_size, num_modalities).
        labels (torch.Tensor): True labels, shape (batch_size,).
        strategy (str): Strategy to use for computing bias weight. Options are 'MinStrategy' or 'AvgStrategy'.

    Returns:
        torch.Tensor: Bias weights for each sample, shape (batch_size,).
    """
    abs_diffs = torch.abs(preds - labels.unsqueeze(1))  # Calculate absolute differences

    if strategy == 'MinStrategy':
        min_abs_diffs = torch.min(abs_diffs, dim=1)[0]  # Minimum absolute difference across modalities
        bias_weights = 1 / min_abs_diffs  # Calculate bias weights
    elif strategy == 'AvgStrategy':
        avg_abs_diffs = torch.mean(abs_diffs, dim=1)  # Average of absolute differences across modalities
        bias_weights = 1 / avg_abs_diffs  # Calculate bias weights
    else:
        raise ValueError("Invalid strategy. Choose 'MinStrategy' or 'AvgStrategy'.")

    return bias_weights

class IPWEnhancedMAELoss(torch.nn.Module):
    def __init__(self):
        super(IPWEnhancedMAELoss, self).__init__()

    def forward(self, mae_loss, bias_weight):
        """
        Compute IPW-enhanced MAE loss.

        Args:
            mae_loss (torch.Tensor): MAE loss without weighting.
            bias_weight (torch.Tensor): Bias weight for each sample.

        Returns:
            torch.Tensor: IPW-enhanced MAE loss.
        """
        return mae_loss * (1 / (bias_weight + 1))

def denormalize(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return np.array(img.cpu().permute(1,2,0)*255).astype(np.uint8)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)