
import os
import argparse
import random
import numpy as np
import json
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset import CTPDataset, ToTensor

from utils.test_util import *
from networks.ed import create_model_ED
from networks.f import create_model_F


parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='DATA', help='Dataset location')
parser.add_argument('--experiment_name', type=str,
                    default="CTP", help='Which experiment to load')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--load_iter', type=str,  default='latest', help='iteration to load from saved snapshot')

parser.add_argument('--patch_x', type=int,
                    default=128, help='H')
parser.add_argument('--patch_y', type=int,
                    default=128, help='W')
parser.add_argument('--patch_z', type=int,
                    default=64, help='T')



parser.add_argument('--stride_xy', type=int,
                    default=32, help='stride to slide the patchwise inference of whole image')
parser.add_argument('--stride_z', type=int,
                    default=16, help='stride to slide the patchwise inference of whole image')

args = parser.parse_args()

# seed everything
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Initialize paths
data_path = args.root_path
experiment_output_path = args.experiment_name

if not os.path.exists(experiment_output_path):
    raise FileNotFoundError('experiment does not exist: {}'.format(os.path.basename(experiment_output_path)))

snapshot_path = os.path.join(experiment_output_path,'ckpt')

test_result_save_path = os.path.join(experiment_output_path, 'test_result')
os.makedirs(test_result_save_path, exist_ok=True)


patch_size = (args.patch_x,args.patch_y,args.patch_z)
stride_xy, stride_z = args.stride_xy, args.stride_z
load_iter = args.load_iter



ED_decoder_branches = 5
ED_input_channels = args.patch_z
F_input_channels = ED_decoder_branches -1


if __name__ == '__main__':

    model_ED = create_model_ED(input_channel = ED_input_channels, 
                               n_filters = 16,
                               decoder_branches = ED_decoder_branches)
    model_F = create_model_F(input_channel = F_input_channels,
                             n_filters = 16)


    save_ED_path = os.path.join(
        snapshot_path, 'ED_iter_' + str(load_iter))

    save_F_path = os.path.join(
        snapshot_path, 'F_iter_' + str(load_iter))

    print("init model ED weight from {}".format(save_ED_path))
    print("init modle F weight from {}".format(save_F_path))

    model_ED.load_state_dict(torch.load(save_ED_path))
    model_F.load_state_dict(torch.load(save_F_path))

    model_ED.eval()
    model_F.eval()

    db_test = CTPDataset(base_dir=data_path,
                    split='validation',  # train/validation split
                    transform=transforms.Compose([
                        ToTensor()
                    ]))     # Random Cropping NOT applied, returns fullsize image

    testloader = DataLoader(db_test, batch_size=1, shuffle=False)

    total_pred_metrics_public = {k:[] for k in ['seg', 'pseg', 'tmax', 'mtt', 'cbv', 'cbf']}
    total_pred_metrics_private = {k:[] for k in ['tmax', 'mtt', 'cbv', 'cbf']}


    for _, sampled_batch in tqdm(enumerate(testloader)):

        single_pred_collection, single_pred_metrics = test_single_case(model_ED, model_F, sampled_batch, stride_xy, stride_z, patch_size)

        if sampled_batch['isPublic'][0]:
            for k in total_pred_metrics_public.keys():
                if single_pred_metrics[k]:  # if not None, for empty seg
                    total_pred_metrics_public[k].append(single_pred_metrics[k])
        else:
            for k in total_pred_metrics_private.keys():
                print(single_pred_metrics[k])
                total_pred_metrics_private[k].append(single_pred_metrics[k])

        save_result(sampled_batch, single_pred_collection, test_result_save_path)


    for k, v in total_pred_metrics_public.items():
        if len(v) > 0:
            total_pred_metrics_public[k] = list(np.mean(v, axis=0))

    for k, v in total_pred_metrics_private.items():
        if len(v) > 0:
            total_pred_metrics_private[k] = list(np.mean(v, axis=0))


    print(total_pred_metrics_public)

    with open('{}/public_data_test_result.txt'.format(test_result_save_path),'w') as f:
        json.dump(total_pred_metrics_public, f, indent=2)
    with open('{}/private_data_test_result.txt'.format(test_result_save_path),'w') as f:
        json.dump(total_pred_metrics_private, f, indent=2)

    
        
                
        
            









