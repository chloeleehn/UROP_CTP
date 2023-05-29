from skimage.io import imsave
import os
import numpy as np
import skimage.metrics as sm
import math
import torch

from utils.losses import dice_score
from utils.util import to255

def save_result(sampled_batch, pred_collection, test_save_path):
    item_path = os.path.join(test_save_path, sampled_batch['name'][0])
    if not os.path.exists(item_path):
        os.makedirs(item_path)

    for item in ['tmax','mtt','cbv','cbf','seg','pseg']:
        imsave(os.path.join(item_path, 'pred_{}.png'.format(item)), to255(pred_collection['pred_{}'.format(item)]))

        if (item != 'pseg' and (item != 'seg' or sampled_batch['labeled'][0])):
            imsave(os.path.join(item_path, 'gt_{}.png'.format(item)), to255(sampled_batch[item].numpy().squeeze()))


def calculate_metric_seg(pred, gt):
    gt = gt.numpy().squeeze()
    if np.any(gt):
        dc = dice_score(pred,gt)
        return dc,   # retuns tuple
    else:
        return None
    

def calculate_metric_map(pred, gt):
    gt = gt.numpy().squeeze()
    ssim = sm.structural_similarity(gt, pred)
    psnr = sm.peak_signal_noise_ratio(gt, pred, data_range=1.0)
    mi = sm.normalized_mutual_information(gt, pred)
    return ssim, psnr, mi       # returns tuple


def tensor2np2D(tensor):
    return tensor.cpu().data.numpy().squeeze()

def test_single_case(model_ED, model_F, sampled_batch, stride_xy, stride_z, patch_size):

    data = sampled_batch['data']

    b, dd, ww, hh = data.shape
    p0,p1,p2 = patch_size

    sx = math.ceil((ww - p0) / stride_xy) + 1
    sy = math.ceil((hh - p1) / stride_xy) + 1
    sz = math.ceil((dd - p2) / stride_z) + 1
    
    empty_arr = np.zeros([ww,hh]).astype(np.float32)
    cnt = np.zeros([ww,hh]).astype(np.float32)
    pred_collection = {k:empty_arr.copy() for k in ['pred_seg', 'pred_pseg', 'pred_tmax', 'pred_mtt', 'pred_cbv', 'pred_cbf']}

    for x in range(0,sx):
        xs = min(stride_xy*x, ww-p0)
        for y in range(0,sy):
            ys = min(stride_xy * y, hh-p1)
            for z in range(0,sz):
                zs = min(stride_z * z, dd-p2)

                patch = data[:, zs:zs+p2, xs:xs+p0, ys:ys+p1].cuda()

                with torch.no_grad():
                    out_dict, fuse_feature = model_ED(patch)
                    out_seg_ = out_dict['out_seg']

                    model_F_input = torch.cat((
                            out_dict['out_map_tmax'] + out_seg_,
                            out_dict['out_map_mtt'] + out_seg_,
                            out_dict['out_map_cbv'] + out_seg_,
                            out_dict['out_map_cbf'] + out_seg_
                        ),1)

                    out_pseg_ = model_F(model_F_input, fuse_feature)

                    out_seg = torch.sigmoid(out_seg_)
                    out_pseg = torch.sigmoid(out_pseg_)

                pred_collection['pred_seg'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_seg)
                pred_collection['pred_pseg'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_pseg)
                pred_collection['pred_tmax'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['out_map_tmax'])
                pred_collection['pred_mtt'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['out_map_mtt'])
                pred_collection['pred_cbv'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['out_map_cbv'])
                pred_collection['pred_cbf'][xs:xs+p0, ys:ys+p1] += tensor2np2D(out_dict['out_map_cbf'])
                cnt[xs:xs+p0, ys:ys+p1] += 1
    
    
    for k,v in pred_collection.items():
        pred_collection[k] = v/cnt

    pred_metrics = {
        'seg':calculate_metric_seg(pred_collection['pred_seg'], sampled_batch['seg']) if sampled_batch['labeled'] else None,
        'pseg':calculate_metric_seg(pred_collection['pred_pseg'], sampled_batch['seg']) if sampled_batch['labeled']  else None,
        'tmax':calculate_metric_map(pred_collection['pred_tmax'], sampled_batch['tmax']),
        'mtt':calculate_metric_map(pred_collection['pred_mtt'], sampled_batch['mtt']),
        'cbv':calculate_metric_map(pred_collection['pred_cbv'], sampled_batch['cbv']),
        'cbf':calculate_metric_map(pred_collection['pred_cbf'], sampled_batch['cbf'])
    }

    return pred_collection, pred_metrics


