import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5


    
    # # Check and transform
    # output_ = force_binary_matrix(output_)
    # target_ = force_binary_matrix(target_)

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    # Example usage

    # hd_ = hd(output_, target_)
    # hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, recall_, specificity_, precision_


def has_binary_object_and_transform(array):
    """
    Check if the array contains any binary object (values of True).
    If it does, convert it to a binary array (0 and 1).
    """
    # Convert the array to boolean to ensure consistency
    binary_array = np.asarray(array, dtype=bool)
    
    # Check if there is any binary object (any True values)
    has_object = np.any(binary_array)
    
    # Convert the boolean array to integers (0 and 1)
    transformed_array = binary_array.astype(np.uint8)
    
    return transformed_array

def force_binary_matrix(array):
    """
    Ensures the array is binary (0 and 1). Copies and converts the input array.
    Parameters:
        array: Input array, can contain any values (e.g., boolean, integers, floats).
    Returns:
        A binary array (values of 0 and 1 only).
    """
    # Convert to a boolean mask (True/False)
    binary_mask = np.asarray(array, dtype=bool)
    
    # Convert boolean to binary (0 and 1)
    binary_array = binary_mask.astype(np.uint8)
    
    return binary_array