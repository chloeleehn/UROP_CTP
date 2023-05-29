import numpy as np
import torch


def to255(data):
    # maxval = 1.0
    # minval = 0.0
    # return (((data - minval) / (maxval - minval))*255).astype(np.uint8)
    return (data*255.0).round().astype(np.uint8)

def to255_t(data):  # from tensor to numpy
    data = np.array(data)
    # maxval = 1.0
    # minval = 0.0
    # return (((data - minval) / (maxval - minval))*255).astype(np.uint8)
    return (data*255.0).round().astype(np.uint8)



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))



def get_current_consistency_weight(epoch, totalepoch, rampmax = 1.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    assert rampmax <= 1.0

    end_epoch = int(totalepoch * rampmax)
    if epoch > end_epoch:
        return 1.0
    else:
        return 1.0 * sigmoid_rampup(epoch, end_epoch)


# def get_domain_num(name_batch):

#     names_num = []

#     for n in name_batch:
#         if 'stanford' in n:
#             names_num.append([1])
#         else:
#             names_num.append([0])

#     return torch.from_numpy(np.array(names_num)).float().cuda()

def get_domain_num(domain_batch):

    domains_num = []

    for isPublic in domain_batch:
        if isPublic: 
            domains_num.append([0])
        else:
            domains_num.append([1])

    return torch.from_numpy(np.array(domains_num)).float().cuda()