import os
import torch
import numpy as np
import random
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from skimage.measure import regionprops, label, find_contours
#import math


class CTPDataset(Dataset):
    """ CTP Dataset """
<<<<<<< HEAD

    def __init__(self, base_dir=None, split='train', transform=None, hon=False, res=1):
=======
    def __init__(self, base_dir=None, split='train', transform=None, hon=False, res = 1):
>>>>>>> origin/main
        self.transform = transform
        self.labeled_list = []
        self.unlabeled_list = []
        self.total_list = []
        self.split = split

        if split == 'train':
            self.labeled_list = sorted(
                glob(os.path.join(base_dir, 'labeled', '*')))
            self.unlabeled_list = sorted(
                glob(os.path.join(base_dir, 'unlabeled', '*')))

        elif split == 'validation':
            self.labeled_list = sorted(
                glob(os.path.join(base_dir, 'labeled_val', '*')))
            self.unlabeled_list = sorted(
                glob(os.path.join(base_dir, 'unlabeled_val', '*')))

        label_temp = []
        unlabel_temp = []
        
        # Iterate through different starting indices
        for i in range(res):
            # Select elements from the original list with the specified step size
            selected_label = self.labeled_list[i::res]
            selected_unlabel = self.unlabeled_list[i::res]
            
            # Concatenate the selected elements with the result list
            label_temp.extend(selected_label)
            unlabel_temp.extend(selected_unlabel)

        self.labeled_list = [item for item in label_temp for _ in range(res)]
        self.unlabeled_list = [item for item in unlabel_temp for _ in range(res)]
        # self.unlabeled_list = unlabel_temp.repeat_interleave(res)

        print("total {} labeled samples".format(len(self.labeled_list)))
        print("total {} unlabeled samples".format(len(self.unlabeled_list)))

        self.total_list = self.labeled_list + self.unlabeled_list

    def __len__(self):

        if self.split == 'train':
            # we need separate lengths of labeled and unlabeled data for training
            return (len(self.labeled_list), len(self.unlabeled_list))
        else:
            return len(self.total_list)

    def __getitem__(self, idx):
        image_path = self.total_list[idx]

        img_name = os.path.basename(image_path)
        isPublic = False if 'stanford_SS' in image_path else True

        h5f = h5py.File(image_path, 'r')

        data = np.array(h5f['slices_over_time'], dtype=np.float32)
        try:
            assert np.any(data)
        except:
            raise AssertionError('EMPTY DATA: {}'.format(img_name))

        tmax = np.array(h5f['tmax'], dtype=np.float32)
        mtt = np.array(h5f['mtt'], dtype=np.float32)
        cbv = np.array(h5f['cbv'], dtype=np.float32)
        cbf = np.array(h5f['cbf'], dtype=np.float32)

        # Tmax range of stanford : 0~250
        # Tmax range of ISLES : 0~25
        # MTT range of stanford : 0~200
        # MTT range of ISLES : 0~15
        # CBV range : Both 0~100
        # CBF range : Both 0~400

        if not isPublic:
            tmax /= 250
            mtt /= 200

        if isPublic:
            tmax /= 25
            mtt /= 15

        cbv /= 100
        cbf /= 400

        assert np.max(tmax) <= 1.0 and np.min(tmax) >= 0
        assert np.max(mtt) <= 1.0 and np.min(mtt) >= 0
        assert np.max(cbv) <= 1.0 and np.min(cbv) >= 0
        assert np.max(cbf) <= 1.0 and np.min(cbf) >= 0

        labeled = False
        try:
            seg = np.array(h5f['seg'], dtype=np.float32)
            labeled = True
        except:
            # No seg label, Empty dummy label
            seg = np.zeros_like(data[:, :, 0])

        sample = {'data': data,
                  'tmax': tmax,
                  'mtt': mtt,
                  'cbv': cbv,
                  'cbf': cbf,
                  'labeled': labeled,
                  'name': img_name,
                  'seg': seg,
                  'isPublic': isPublic}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, offset_magnitude_ratio):
        self.output_size = output_size
        self.offset_magnitude_ratio = offset_magnitude_ratio

    def cut(self, data, tmax, mtt, cbv, cbf, seg, sizes):

        w1, h1, d1 = sizes

        data = data[h1:h1 + self.output_size[1], w1:w1 +
                    self.output_size[0], d1:d1 + self.output_size[2]]
        tmax = tmax[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]
        mtt = mtt[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]
        cbv = cbv[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]
        cbf = cbf[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]
        seg = seg[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]

        return data, tmax, mtt, cbv, cbf, seg

    def __call__(self, sample):
        data, tmax, mtt, cbv, cbf, seg = sample['data'], sample[
            'tmax'], sample['mtt'], sample['cbv'], sample['cbf'], sample['seg']
        labeled = sample['labeled']

        (w, h, d) = data.shape

        while True:

            legion_centered_patch = False
            if labeled:
                if np.any(seg):
                    if random.random() > 0.3:
                        legion_centered_patch = True

            if legion_centered_patch:
                contours = find_contours(np.array(seg), 0.5)
                num_cont = len(contours)
                assert num_cont > 0

                cont = random.choice(contours)
                y_min, y_max, x_min, x_max = np.min(cont[:, 0]), np.max(
                    cont[:, 0]), np.min(cont[:, 1]), np.max(cont[:, 1])

                if self.offset_magnitude_ratio > 0:
                    x_offset = random.randint(-self.output_size[0] // self.offset_magnitude_ratio,
                                              self.output_size[0] // self.offset_magnitude_ratio)
                    y_offset = random.randint(-self.output_size[1] // self.offset_magnitude_ratio,
                                              self.output_size[1] // self.offset_magnitude_ratio)
                else:
                    x_offset = 0
                    y_offset = 0
                patch_center = ((x_max-x_min) // 2 + x_min +
                                x_offset, (y_max-y_min) // 2 + y_min + y_offset)

                w1 = int(patch_center[0] - self.output_size[0] // 2)
                h1 = int(patch_center[1] - self.output_size[1] // 2)

                if w1 < 0:
                    w1 = 0
                elif w1 > w-self.output_size[0]:
                    w1 = w-self.output_size[0]

                if h1 < 0:
                    h1 = 0
                elif h1 > h-self.output_size[1]:
                    h1 = h-self.output_size[1]

            else:
                w1 = np.random.randint(0, w - self.output_size[0])
                h1 = np.random.randint(0, h - self.output_size[1])

            d1 = 0 if d <= self.output_size[2] else np.random.randint(
                0, d - self.output_size[2])

            data_cut, tmax_cut, mtt_cut, cbv_cut, cbf_cut, seg_cut = self.cut(
                data, tmax, mtt, cbv, cbf, seg, (w1, h1, d1))

            if np.any(data_cut):
                break

        new_sample = sample.copy()
        new_sample['data'] = data_cut
        new_sample['tmax'] = tmax_cut
        new_sample['mtt'] = mtt_cut
        new_sample['cbv'] = cbv_cut
        new_sample['cbf'] = cbf_cut
        new_sample['seg'] = seg_cut

        return new_sample


class RandomFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, flip_p):
        self.flip_p = flip_p

    def __call__(self, sample):

        if random.random() <= self.flip_p:

            data, tmax, mtt, cbv, cbf = sample['data'], sample['tmax'], sample['mtt'], sample['cbv'], sample['cbf']
            new_sample = sample.copy()

            data = np.flip(data, axis=1)
            tmax = np.flip(tmax, axis=1)
            mtt = np.flip(mtt, axis=1)
            cbv = np.flip(cbv, axis=1)
            cbf = np.flip(cbf, axis=1)

            new_sample['data'] = data
            new_sample['tmax'] = tmax
            new_sample['mtt'] = mtt
            new_sample['cbv'] = cbv
            new_sample['cbf'] = cbf

            if sample['labeled']:
                seg = sample['seg']
                seg = np.flip(seg, axis=1)
                new_sample['seg'] = seg

            return new_sample

        else:

            return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, tmax, mtt, cbv, cbf, seg = sample['data'], sample[
            'tmax'], sample['mtt'], sample['cbv'], sample['cbf'], sample['seg']

        data = data.transpose((2, 0, 1)).copy()
        # data = data[np.newaxis, :, :, :].copy()
        tmax = tmax[np.newaxis, :, :].copy()
        mtt = mtt[np.newaxis, :, :].copy()
        cbv = cbv[np.newaxis, :, :].copy()
        cbf = cbf[np.newaxis, :, :].copy()
        seg = seg[np.newaxis, :, :].copy()

        data = data/80.0  # rescale input to 0~1
        data = (data-0.135) / 0.25  # normalize with mean&std

        new_sample = sample.copy()
        new_sample['data'] = torch.from_numpy(data)
        new_sample['tmax'] = torch.from_numpy(tmax)
        new_sample['mtt'] = torch.from_numpy(mtt)
        new_sample['cbv'] = torch.from_numpy(cbv)
        new_sample['cbf'] = torch.from_numpy(cbf)
        new_sample['seg'] = torch.from_numpy(seg)

        return new_sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
"""

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, labeled_is_primary):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        self.labeled_is_primary = labeled_is_primary

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):

        if self.labeled_is_primary:

            primary_iter = iterate_once(self.primary_indices)
            secondary_iter = iterate_eternally(self.secondary_indices)

        else:
            primary_iter = iterate_eternally(self.primary_indices)
            secondary_iter = iterate_once(self.secondary_indices)

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
