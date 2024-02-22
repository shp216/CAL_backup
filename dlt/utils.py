import colorsys
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    batch_data = [{k: v for k, v in item.items() if k != 'ids'} for item in batch]
    ids = [item['ids'] for item in batch]  # 'ids' 수집
    batch_collated = torch.utils.data.dataloader.default_collate(batch_data)
    return batch_collated, ids

def HSVToRGB(h, s, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * r), int(255 * g), int(255 * b)


def getDistinctColors(n):
    huePartition = 1.0 / (n + 1)
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))


colors_f = getDistinctColors(25)
colors = []
for c in colors_f:
    colors.append(c)


def masked_acc(real, pred, mask):
    accuracies = torch.logical_and(real.eq(torch.argmax(pred, dim=-1)), mask)
    return accuracies.sum() / mask.sum()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_cross_entropy(a, b, mask):
    b_c = torch.nn.functional.one_hot(b, num_classes=a.shape[-1])
    a_c = F.log_softmax(a, dim=2)

    loss = (-a_c * b_c).sum(axis=2)
    non_zero_elements = sum_flat(mask)
    loss = sum_flat(loss * mask.float())
    loss = loss / (non_zero_elements + 0.0001)
    return loss


def masked_l2(a, b, mask):
    
    a = a[:, :, :4]
    b = b[:, :, :4]
    mask = mask[:, :, :4]
    # print("###############################################################")
    # print("a.shape: ", a.shape)
    # print("b.shape: ", b.shape)
    # print("mask.shape: ", mask.shape)
    # print("###############################################################")
    loss = F.mse_loss(a, b, reduction='none')
    loss = sum_flat(loss * mask.float())
    non_zero_elements = sum_flat(mask)

    mse_loss_val = (non_zero_elements > 0) * (loss / (non_zero_elements + 0.00000001))
    return mse_loss_val

def masked_l2_r(a, b, mask):
    # 첫 4개 요소에 대한 MSE 손실 계산
    mse_loss_bbox = F.mse_loss(a[:, :, :4], b[:, :, :4], reduction='none') * mask[:, :, :4].float()
    mse_loss_bbox = sum_flat(mse_loss_bbox) / (sum_flat(mask[:, :, :4]) + 1e-8)
    
    # a의 5번째 요소와 b의 6번째 요소에 대한 MSE 계산
    mse_loss_z = F.mse_loss(a[:, :, 5].unsqueeze(-1), b[:, :, 6].unsqueeze(-1), reduction='none') * mask[:, :, 5].unsqueeze(-1).float()
    mse_loss_z = sum_flat(mse_loss_z) / (sum_flat(mask[:, :, 5].unsqueeze(-1)) + 1e-8)
    
    # a의 4번째 요소를 cos과 sin으로 변환하여 b의 4,5번째 요소와 비교
    a_cos = torch.cos(a[:, :, 4] * 2 * torch.pi)
    a_sin = torch.sin(a[:, :, 4] * 2 * torch.pi)
    mse_loss_cos = F.mse_loss(a_cos.unsqueeze(-1), b[:, :, 4].unsqueeze(-1), reduction='none') * mask[:, :, 4].unsqueeze(-1).float()
    mse_loss_sin = F.mse_loss(a_sin.unsqueeze(-1), b[:, :, 5].unsqueeze(-1), reduction='none') * mask[:, :, 4].unsqueeze(-1).float()
    mse_loss_r = (sum_flat(mse_loss_cos) + sum_flat(mse_loss_sin)) / (sum_flat(mask[:, :, 4].unsqueeze(-1)) + 1e-8)
    
    # b의 4번째 및 5번째 요소의 제곱합이 1이 되도록 하는 손실 추가
    norm_loss = ((b[:, :, 4]**2 + b[:, :, 5]**2 - 1)**2).mean()
    
    # 전체 손실 합산
    total_loss = mse_loss_bbox + mse_loss_z*0.1 + mse_loss_r*0.1 + norm_loss*0.01
    return total_loss

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def plot_sample(box, classes, z_indexes, color_mapping_dict, width=360, height=360):
    thickness = -1
    canvas = (255 * np.ones((height, width, 3))).astype('uint8')
    # sort boxes and classes by z_index
    if z_indexes:
        sort_ixs = np.argsort(z_indexes)
        box = box[sort_ixs]
        classes = classes[sort_ixs]

    for ii, (t, c) in enumerate(zip(box, classes)):
        xs = int((t[0] - t[2] / 2) * width)
        ys = int((t[1] - t[3] / 2) * height)
        xe = int((t[0] + t[2] / 2) * width)
        ye = int((t[1] + t[3] / 2) * height)
        canvas = cv2.rectangle(canvas, (xs, ys), (xe, ye), color=color_mapping_dict[c], thickness=thickness)
        cv2.rectangle(canvas, (xs, ys), (xe, ye), (255, 255, 255), 2)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


def draw_layout_opacity(box, classes, z_indexes, color_mapping_dict, width=360, height=360, opacity=0.8):
    canvas = (255 * np.ones((height, width, 3))).astype('uint8')
    # sort boxes and classes by z_index
    if z_indexes:
        sort_ixs = np.argsort(z_indexes)
        box = box[sort_ixs]
        classes = classes[sort_ixs]

    for ii, (t, c) in enumerate(zip(box, classes)):

        xs = int((t[0] - t[2] / 2) * width)
        ys = int((t[1] - t[3] / 2) * height)
        xe = int((t[0] + t[2] / 2) * width)
        ye = int((t[1] + t[3] / 2) * height)

        overlay = canvas.copy()

        canvas = cv2.rectangle(canvas, (xs, ys), (xe, ye), color=color_mapping_dict[c], thickness=-1)
        canvas = cv2.addWeighted(overlay, opacity, canvas, 1 - opacity, 0)
        canvas = cv2.rectangle(canvas, (xs, ys), (xe, ye), color=color_mapping_dict[c], thickness=2)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas


