import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import json
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

# Losses
from utils.losses import FocalLoss, BCEDicePenalizeBorderLoss
from torchmetrics import StructuralSimilarityIndexMeasure
from torch.nn import BCEWithLogitsLoss, MSELoss


from utils.util import get_current_consistency_weight, get_domain_num, to255_t
from datasets.dataset import *


from networks.c import *
from networks.ed import create_model_ED
from networks.f import create_model_F
from datetime import datetime



from skimage.io import imsave



parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='DATA', help='Dataset location')
parser.add_argument('--base_experiment_name', type=str,
                    default="CTP", help='Where to save expreiment results')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--patch_x', type=int,
                    default=128, help='H')
parser.add_argument('--patch_y', type=int,
                    default=128, help='W')
parser.add_argument('--patch_z', type=int,
                    default=64, help='T')


parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum iterations to train')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size per gpu')

parser.add_argument('--labeled_bs', type=int, default=32,
                    help='labeled_batch_size per gpu')



parser.add_argument('--lr_ed', type=float,  default=3e-3,
                    help='starting lr for model ED')
parser.add_argument('--lr_f', type=float,  default=5e-3,
                    help='starting lr for model F')
parser.add_argument('--lr_c', type=float,  default=1e-3,
                    help='starting lr for model C')



parser.add_argument('--segloss', type=str,
                    default='dicece', help='ce | dicece | focal  Loss for segmentation')
parser.add_argument('--rampmax', type=float,
                    default=0.75, help='When will the strength of consistency loss reach its max (1.0 = at the end of training)')
parser.add_argument('--gamma', type=float,  default=0.8,
                    help='balance factor to control supervised fulldata map loss and supervised partiallabel seg loss')


parser.add_argument('--crop_offset', type=int,  default=5, help='x, random offset = H/x for lesion-centered patching')
parser.add_argument('--flip_prob', type=float,  default=0.5, help='probability of horizontal flip')
parser.add_argument('--dice_ratio', type=float,  default=0.5, help='if segloss=dicece, ration between dice and ce')


parser.add_argument('--log_loss_freq', type=int,  default=100, help='how frequently report loss')
parser.add_argument('--print_sample_freq', type=int,  default=2000, help='how frequently save sample training results')
parser.add_argument('--save_ckpt_freq', type=int,  default=2000, help='how frequently save model checkpoints')

parser.add_argument('--loglevel', choices=['debug', 'info', 'critical'], default='info', help='Log level')
args = parser.parse_args()

log_mapping = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'critical': logging.CRITICAL,
}
logging.basicConfig(encoding='utf-8', level=log_mapping[args.loglevel])
logging.info(str(args))

# Seed everything
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# GPU to use, sinlge gpu only (TODO implement multi-gpu support)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Initialize paths to save experiment results
data_path = args.root_path
experiment_output_path = args.base_experiment_name + '_{}'.format(datetime.now().strftime('%y%m%d_%H%M%S'))
os.makedirs(experiment_output_path, exist_ok=True)

with open('{}/options.txt'.format(experiment_output_path),'w') as f:
    json.dump(args.__dict__, f, indent=2)

snapshot_path = os.path.join(experiment_output_path,'ckpt')
os.makedirs(snapshot_path, exist_ok=True)

sample_output_path = os.path.join(experiment_output_path, 'sample_train_output')
os.makedirs(sample_output_path, exist_ok=True)


# dimension of input 3D CTP volume - H,W,T
patch_size = (args.patch_x,args.patch_y,args.patch_z)

# batch size, labeled + unlabeled
batch_size = args.batch_size * len(args.gpu.split(','))
labeled_bs = args.labeled_bs
unlabeled_bs = batch_size - labeled_bs

# Total # of iterations to run
max_iterations = args.max_iterations

# # After every 40% of the iteration, half the lr
# lr_dec_freq = int(max_iterations) * 0.4 

# starting lrs
base_lr_ED = args.lr_ed
base_lr_F = args.lr_f
base_lr_C = args.lr_c

# 4 parameter maps + 1 segmentation
ED_decoder_branches = 5  
# Channel axis = T
ED_input_channels = args.patch_z
# 4 predicted parameter maps as inputs to F
F_input_channels = ED_decoder_branches -1




if __name__ == "__main__":

    model_ED = create_model_ED(input_channel = ED_input_channels, 
                               n_filters = 16,
                               decoder_branches = ED_decoder_branches)
    model_F = create_model_F(input_channel = F_input_channels,
                             n_filters = 16)
    model_C = AdversarialNetwork(max_iterations).cuda()  # Domain adaptation module
    # https://arxiv.org/abs/1505.07818 - Domain-Adversarial training of neural networks

    db_train = CTPDataset(base_dir=data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomFlip(args.flip_prob),
                           RandomCrop(patch_size, args.crop_offset),
                           ToTensor(),
                       ]))



    labeled_num, unlabeled_num =  db_train.__len__()

    labeled_idxs = range(labeled_num)
    unlabeled_idxs = range(labeled_num, labeled_num + unlabeled_num)
    
    # Randomly samples from both labeled and unlabeled dataset such that 
    # in each batch, first *labeled_bs* data are labeled and remaining *batch_size - labeled_bs* data are unlabeled 
    batch_sampler = TwoStreamBatchSampler(
        primary_indices = labeled_idxs, 
        secondary_indices = unlabeled_idxs, 
        batch_size = batch_size, 
        secondary_batch_size = batch_size-labeled_bs, 
        labeled_is_primary = True)


    # def worker_init_fn(worker_id):
    #     random.seed(args.seed+worker_id)
        
    # trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
    #                          num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler)


    model_ED.train()
    model_F.train()
    model_C.train()
    
    optimizer_ED = optim.Adam(model_ED.parameters(), lr=base_lr_ED, betas=(0.5, 0.99))
    optimizer_F = optim.Adam(model_F.parameters(), lr = base_lr_F, betas=(0.5, 0.99))
    optimizer_C = optim.Adam(model_C.parameters(), lr = base_lr_C, betas=(0.5, 0.99))

    
    foc_loss = FocalLoss()
    dicece = BCEDicePenalizeBorderLoss(dice_ratio=args.dice_ratio)  # dice + crossentropy
    ce_loss = BCEWithLogitsLoss() # crossentropy
    mse_loss =  MSELoss()
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    

    if args.segloss == 'focal':
        seg_loss_fn = foc_loss
    elif args.segloss == 'dicece':
        seg_loss_fn = dicece
    else:
        seg_loss_fn = ce_loss

    def map_loss_compute(pred_map, gt_map):  # MSE + 1-SSIM 
        ssim_loss_component = 1-ssim_loss(pred_map, gt_map)
        mse_loss_component = mse_loss(pred_map, gt_map)
        return ssim_loss_component + mse_loss_component 

    logging.info("{} itertations per epoch".format(len(trainloader)))


    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    iterator = tqdm(range(max_epoch), ncols=70)
    writer = SummaryWriter('{}/log'.format(experiment_output_path))

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):

            labeled_flag = sampled_batch['labeled']
            assert torch.all(labeled_flag[:labeled_bs]) # All are labeled
            assert not torch.any(labeled_flag[labeled_bs:]) # All are unlabeled

            # labeled data load
            data_batch_labeled = sampled_batch['data'][:labeled_bs].cuda()
            tmax_batch_labeled = sampled_batch['tmax'][:labeled_bs].cuda()      
            mtt_batch_labeled = sampled_batch['mtt'][:labeled_bs].cuda()
            cbv_batch_labeled = sampled_batch['cbv'][:labeled_bs].cuda()
            cbf_batch_labeled = sampled_batch['cbf'][:labeled_bs].cuda()
            seg_batch = sampled_batch['seg'][:labeled_bs].cuda()

            # unlabeled data load
            data_batch_unlabeled = sampled_batch['data'][labeled_bs:].cuda()
            tmax_batch_unlabeled = sampled_batch['tmax'][labeled_bs:].cuda()
            mtt_batch_unlabeled = sampled_batch['mtt'][labeled_bs:].cuda()
            cbv_batch_unlabeled = sampled_batch['cbv'][labeled_bs:].cuda()
            cbf_batch_unlabeled = sampled_batch['cbf'][labeled_bs:].cuda()

            # strength of L_cons weight. 
            consistency_weight = get_current_consistency_weight(epoch_num, max_epoch, args.rampmax)

            # Start Alternating training - One pass on labeled batch, following pass on unlableded batch
            # Start labeled batch ###############################################################################################################################################################################################################################################################

            out_dict_lab, fuse_feat_lab = model_ED(data_batch_labeled)
            bottleneck_feature_lab = fuse_feat_lab[-1]

            out_seg_lab = out_dict_lab['out_seg']
            out_map_tmax_lab, out_map_mtt_lab, out_map_cbv_lab, out_map_cbf_lab = \
                out_dict_lab['out_map_tmax'],out_dict_lab['out_map_mtt'],out_dict_lab['out_map_cbv'],out_dict_lab['out_map_cbf']
            
            # fuse predicted seg to input to F
            model_F_input_lab = torch.cat((
                    out_map_tmax_lab + out_seg_lab,
                    out_map_mtt_lab + out_seg_lab,
                    out_map_cbv_lab + out_seg_lab,
                    out_map_cbf_lab + out_seg_lab,
                ),1)
            
            # predicted pseudo seg mask
            out_pseudo_seg_lab = model_F(model_F_input_lab, fuse_feat_lab)


            # Supervised segmentation loss - only for labeled batch where GT seg is available
            loss_seg = seg_loss_fn(out_seg_lab, seg_batch)
            loss_pseudo_seg = seg_loss_fn(out_pseudo_seg_lab, seg_batch)
            

            # Map estimation loss
            loss_map_tmax_lab = map_loss_compute(out_map_tmax_lab, tmax_batch_labeled)
            loss_map_mtt_lab = map_loss_compute(out_map_mtt_lab, mtt_batch_labeled)
            loss_map_cbv_lab = map_loss_compute(out_map_cbv_lab, cbv_batch_labeled)
            loss_map_cbf_lab = map_loss_compute(out_map_cbf_lab, cbf_batch_labeled)

            # Consistency Loss
            loss_cons_lab = torch.mean((out_seg_lab - out_pseudo_seg_lab) ** 2) * consistency_weight


            supervised_seg_loss_lab = loss_seg + loss_pseudo_seg 

            supervised_map_loss_lab = (loss_map_tmax_lab+\
                                            loss_map_mtt_lab+\
                                            loss_map_cbv_lab+\
                                            loss_map_cbf_lab) * args.gamma


            # da_domain_arr_lab = get_domain_num(sampled_batch['name'][:labeled_bs])
            da_domain_arr_lab = get_domain_num(sampled_batch['isPublic'][:labeled_bs])

            da_loss_lab = DANN(bottleneck_feature_lab, model_C, da_domain_arr_lab)  
            

            loss_labeled_batch = supervised_seg_loss_lab +\
                                supervised_map_loss_lab +\
                                loss_cons_lab +\
                                da_loss_lab

            optimizer_ED.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()

            loss_labeled_batch.backward()

            optimizer_ED.step()
            optimizer_F.step()  
            optimizer_C.step()

            # Start unlabeled batch ################################################################################################################################################################################################################################################################################


            out_dict_unlab, fuse_feat_unlab = model_ED(data_batch_unlabeled)
            bottleneck_feature_unlab = fuse_feat_unlab[-1]

            out_seg_unlab = out_dict_unlab['out_seg']
            out_map_tmax_unlab, out_map_mtt_unlab, out_map_cbv_unlab, out_map_cbf_unlab = \
                out_dict_unlab['out_map_tmax'],out_dict_unlab['out_map_mtt'],out_dict_unlab['out_map_cbv'],out_dict_unlab['out_map_cbf']

            model_F_input_unlab = torch.cat((
                    out_map_tmax_unlab + out_seg_unlab,
                    out_map_mtt_unlab + out_seg_unlab,
                    out_map_cbv_unlab + out_seg_unlab,
                    out_map_cbf_unlab + out_seg_unlab,
                ),1)


            out_pseudo_seg_unlab = model_F(model_F_input_unlab, fuse_feat_unlab)

            # Map estimation loss
            loss_map_tmax_unlab = map_loss_compute(out_map_tmax_unlab, tmax_batch_unlabeled)
            loss_map_mtt_unlab = map_loss_compute(out_map_mtt_unlab, mtt_batch_unlabeled)
            loss_map_cbv_unlab = map_loss_compute(out_map_cbv_unlab, cbv_batch_unlabeled)
            loss_map_cbf_unlab = map_loss_compute(out_map_cbf_unlab, cbf_batch_unlabeled)

            # Consistency Loss
            loss_cons_unlab = torch.mean((out_seg_unlab - out_pseudo_seg_unlab) ** 2) * consistency_weight

            supervised_map_loss_unlab = (loss_map_tmax_unlab+\
                                            loss_map_mtt_unlab+\
                                            loss_map_cbv_unlab+\
                                            loss_map_cbf_unlab) * args.gamma

            # da_domain_arr_unlab = get_domain_num(sampled_batch['name'][labeled_bs:])
            da_domain_arr_unlab = get_domain_num(sampled_batch['isPublic'][labeled_bs:])
            da_loss_unlab = DANN(bottleneck_feature_unlab, model_C, da_domain_arr_unlab)

            loss_unlabeled_batch = supervised_map_loss_unlab +\
                                    loss_cons_unlab +\
                                    da_loss_unlab

            optimizer_ED.zero_grad()
            # optimizer_F.zero_grad()    
            optimizer_C.zero_grad()

            loss_unlabeled_batch.backward()

            optimizer_ED.step()
            # optimizer_F.step()   # Freeze Model F if the ground truth label is not available!
            optimizer_C.step()
    
            # Pass END #################################################


            # Update learning rate (decay) 
            # if iter_num % lr_dec_freq == 0:
        
            #     lr_ED = base_lr_ED * 0.5 ** (iter_num // lr_dec_freq) 
            #     for param_group in optimizer_ED.param_groups:
            #         param_group['lr'] = lr_ED

            #     lr_F = base_lr_F * 0.5 ** (iter_num // lr_dec_freq)
            #     for param_group in optimizer_F.param_groups:
            #         param_group['lr'] = lr_F

            #     lr_C = base_lr_C * 0.5 ** (iter_num // lr_dec_freq)
            #     for param_group in optimizer_C.param_groups:
            #         param_group['lr'] = lr_C

        
            lr_ED = base_lr_ED * (1.0 - iter_num/max_iterations) ** 0.9 
            for param_group in optimizer_ED.param_groups:
                param_group['lr'] = lr_ED

            lr_F = base_lr_F * (1.0 - iter_num/max_iterations) ** 0.9 
            for param_group in optimizer_F.param_groups:
                param_group['lr'] = lr_F

            lr_C = base_lr_C * (1.0 - iter_num/max_iterations) ** 0.9 
            for param_group in optimizer_C.param_groups:
                param_group['lr'] = lr_C


            # increment iter
            iter_num = iter_num + 1


            # Start Logging ###################################################################

            # Learning rates
            writer.add_scalar('lr/lr_ED',lr_ED, iter_num)
            writer.add_scalar('lr/lr_F',lr_F, iter_num)
            writer.add_scalar('lr/lr_C',lr_C, iter_num)
            # Labeled batch losses
            writer.add_scalar('loss/labeled/seg', loss_seg, iter_num)
            writer.add_scalar('loss/labeled/pseudo_seg', loss_pseudo_seg, iter_num)
            writer.add_scalar('loss/labeled/tmax', loss_map_tmax_lab, iter_num)
            writer.add_scalar('loss/labeled/mtt', loss_map_mtt_lab, iter_num)
            writer.add_scalar('loss/labeled/cbv', loss_map_cbv_lab, iter_num)
            writer.add_scalar('loss/labeled/cbf', loss_map_cbf_lab, iter_num)
            writer.add_scalar('loss/labeled/consistency', loss_cons_lab, iter_num)
            # Unlabeled batch losses
            writer.add_scalar('loss/unlabeled/tmax', loss_map_tmax_unlab, iter_num)
            writer.add_scalar('loss/unlabeled/mtt', loss_map_mtt_unlab, iter_num)
            writer.add_scalar('loss/unlabeled/cbv', loss_map_cbv_unlab, iter_num)
            writer.add_scalar('loss/unlabeled/cbf', loss_map_cbf_unlab, iter_num)
            writer.add_scalar('loss/unlabeled/consistency', loss_cons_unlab, iter_num)
            
            writer.add_scalar('loss/labeled/da', da_loss_lab, iter_num)
            writer.add_scalar('loss/unlabeled/da', da_loss_unlab, iter_num)

            writer.add_scalar('consistency_weight', consistency_weight, iter_num)
                    
            if iter_num % args.log_loss_freq == 0:
                logging.info("EPOCH {} ITER {} consweight {:.3f} | Seg {:.3f} SegPseudo {:.3f} Tmax {:.3f} Mtt {:.3f} Cbv {:.3f} Cbf {:.3f} Cons_lab {:.3f} Cons_unlab {:.3f} Da_lab {:.3f} Da_unlab {:.3f}"\
                            .format(
                                    epoch_num,
                                    iter_num, 
                                    consistency_weight, 
                                    loss_seg.item(), 
                                    loss_pseudo_seg.item(), 
                                    (loss_map_tmax_lab.item() + loss_map_tmax_unlab.item())/2,
                                    (loss_map_mtt_lab.item() + loss_map_mtt_unlab.item())/2,
                                    (loss_map_cbv_lab.item() + loss_map_cbv_unlab.item())/2,
                                    (loss_map_cbf_lab.item() + loss_map_cbf_unlab.item())/2,
                                    loss_cons_lab.item(),
                                    loss_cons_unlab.item(),
                                    da_loss_lab.item(),
                                    da_loss_unlab.item()
                            ))
                

                
            # save intermediate inference results on training data
            if iter_num % args.print_sample_freq == 0:
                for b in range((labeled_bs)):
                    
                    seg = to255_t(sampled_batch['seg'][b,0,:,:])
                    tmax = to255_t(sampled_batch['tmax'][b,0,:,:])
                    mtt = to255_t(sampled_batch['mtt'][b,0,:,:])
                    cbv = to255_t(sampled_batch['cbv'][b,0,:,:])
                    cbf = to255_t(sampled_batch['cbf'][b,0,:,:])

                    out_map_tmax = to255_t(out_map_tmax_lab[b,0,:,:].detach().cpu())
                    out_map_mtt = to255_t(out_map_mtt_lab[b,0,:,:].detach().cpu())
                    out_map_cbv = to255_t(out_map_cbv_lab[b,0,:,:].detach().cpu())
                    out_map_cbf = to255_t(out_map_cbf_lab[b,0,:,:].detach().cpu())

                    out_seg = to255_t(torch.sigmoid(out_seg_lab[b,0,:,:].detach().cpu()))
                    out_seg_pseudo = to255_t(torch.sigmoid(out_pseudo_seg_lab[b,0,:,:].detach().cpu()))

                    seg_montage = np.hstack([seg, out_seg, out_seg_pseudo])
                    map_montage = np.vstack([np.hstack([tmax,out_map_tmax]),np.hstack([mtt,out_map_mtt]),np.hstack([cbv,out_map_cbv]),np.hstack([cbf,out_map_cbf])])

                    imsave('{}/{}_map_lab_{}.png'.format(sample_output_path,iter_num, b), map_montage)
                    imsave('{}/{}_seg_lab_{}.png'.format(sample_output_path,iter_num, b), seg_montage)

                    break

                for b in range((batch_size - labeled_bs)):
                    tmax = to255_t(sampled_batch['tmax'][labeled_bs+b,0,:,:])
                    mtt = to255_t(sampled_batch['mtt'][labeled_bs+b,0,:,:])
                    cbv = to255_t(sampled_batch['cbv'][labeled_bs+b,0,:,:])
                    cbf = to255_t(sampled_batch['cbf'][labeled_bs+b,0,:,:])

                    out_map_tmax = to255_t(out_map_tmax_unlab[b,0,:,:].detach().cpu())
                    out_map_mtt = to255_t(out_map_mtt_unlab[b,0,:,:].detach().cpu())
                    out_map_cbv = to255_t(out_map_cbv_unlab[b,0,:,:].detach().cpu())
                    out_map_cbf = to255_t(out_map_cbf_unlab[b,0,:,:].detach().cpu())

                    out_seg = to255_t(torch.sigmoid(out_seg_unlab[b,0,:,:].detach().cpu()))
                    out_seg_pseudo = to255_t(torch.sigmoid(out_pseudo_seg_unlab[b,0,:,:].detach().cpu()))

                    seg_montage = np.hstack([out_seg, out_seg_pseudo])
                    map_montage = np.vstack([np.hstack([tmax,out_map_tmax]),np.hstack([mtt,out_map_mtt]),np.hstack([cbv,out_map_cbv]),np.hstack([cbf,out_map_cbf])])

                    imsave('{}/{}_map_unlab_{}.png'.format(sample_output_path,iter_num, b), map_montage)
                    imsave('{}/{}_seg_unlab_{}.png'.format(sample_output_path,iter_num, b), seg_montage)

                    break




            if iter_num % args.save_ckpt_freq == 0 or iter_num >= max_iterations:

                checkpoint_identifier = str(iter_num) if iter_num < max_iterations else 'latest'

                save_ED_path = os.path.join(
                    snapshot_path, 'ED_iter_' + checkpoint_identifier)

                save_F_path = os.path.join(
                    snapshot_path, 'F_iter_' + checkpoint_identifier)

                save_C_path = os.path.join(
                    snapshot_path, 'C_iter_' + checkpoint_identifier)
                
                logging.info("saving model to {} at iter {}".format(snapshot_path, iter_num))
                torch.save(model_ED.state_dict(), save_ED_path)
                logging.info("- save model ED to {}".format(save_ED_path))
                torch.save(model_F.state_dict(), save_F_path)
                logging.info("- save model F to {}".format(save_F_path))
                torch.save(model_C.state_dict(), save_C_path)
                logging.info("- save model C to {}".format(save_C_path))


            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
