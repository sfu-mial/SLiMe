import math
import numpy as np
import torch


def adjust_bbox_coords(long_side_length, short_side_length, short_side_coord_min, short_side_coord_max,
                       short_side_original_length):
    margin = (long_side_length - short_side_length) / 2
    short_side_coord_min -= math.ceil(margin)
    short_side_coord_max += math.floor(margin)
    short_side_coord_max -= min(0, short_side_coord_min)
    short_side_coord_min = max(0, short_side_coord_min)
    short_side_coord_min += min(short_side_original_length - short_side_coord_max, 0)
    short_side_coord_max = min(short_side_original_length, short_side_coord_max)

    return short_side_coord_min, short_side_coord_max


def get_square_cropping_coords(mask, square_size=None):
    ys, xs = torch.where(mask == 1)
    x_start, x_end, y_start, y_end = xs.min().item(), xs.max().item() + 1, ys.min().item(), ys.max().item() + 1
    w, h = x_end - x_start, y_end - y_start
    if square_size is not None:
        if w < h:
            # diff = max(square_size, h) - h
            diff = square_size - h
            offset_start, offset_end = math.ceil(diff / 2), math.floor(diff / 2)
            if y_start - math.ceil(diff / 2) < 0:
                offset_end -= y_start - math.ceil(diff / 2)
                offset_start = y_start
            elif y_end + math.floor(diff / 2) > 512:
                offset_start += y_end + math.floor(diff / 2) - 512
                offset_end = 512 - y_end
            y_start -= offset_start
            y_end += offset_end

        elif h < w:
            # diff = max(512 // 2, w) - w
            diff = square_size - w
            offset_start, offset_end = math.ceil(diff / 2), math.floor(diff / 2)
            if x_start - math.ceil(diff / 2) < 0:
                offset_end -= x_start - math.ceil(diff / 2)
                offset_start = x_start
            elif x_end + math.floor(diff / 2) > 512:
                offset_start += x_end + math.floor(diff / 2) - 512
                offset_end = 512 - x_end
            x_start -= offset_start
            x_end += offset_end

    w, h = x_end - x_start, y_end - y_start
    if w > h:
        y_start, y_end = adjust_bbox_coords(w, h, y_start, y_end, 512)
    elif w < h:
        x_start, x_end = adjust_bbox_coords(h, w, x_start, x_end, 512)
    return x_start, x_end, y_start, y_end, max(w, h)


def calculate_iou(prediction, mask):
    intersection = (prediction * mask)
    union = prediction + mask - intersection
    return intersection.sum() / (union.sum() + 1e-7)


def post_process_attention_map(attention_map, target_coords):
    y_start, y_end, x_start, x_end = target_coords
    crop_size = y_end - y_start
    if attention_map.shape[0] != crop_size:
        attention_map = torch.nn.functional.interpolate(attention_map[None, None, ...], crop_size, mode="bilinear")[0, 0]

    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    original_size_attention_map = torch.zeros(512, 512)
    original_size_attention_map[y_start: y_end, x_start: x_end] = attention_map
    # binarized_original_size_attention_map = torch.where(original_size_attention_map > threshold, 1., 0.)
    return original_size_attention_map


def get_crops_coords(image_size, crop_size, num_crops_per_side):
    h, w = image_size
    if num_crops_per_side == 1:
        x_step_size = y_step_size = 0
    else:
        x_step_size = ((w-crop_size)//(num_crops_per_side-1))
        y_step_size = ((h-crop_size)//(num_crops_per_side-1))
    crops_coords = []
    for i in range(num_crops_per_side):
        for j in range(num_crops_per_side):
            y_start, y_end, x_start, x_end = i * y_step_size, i * y_step_size + crop_size, j * x_step_size, j * x_step_size + crop_size
            crops_coords.append([y_start, y_end, x_start, x_end])
    return crops_coords


def get_bbox_data(mask):
    ys, xs = np.where(mask == 1)
    return xs.min(), xs.max(), ys.min(), ys.max()
