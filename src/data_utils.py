from PIL import Image
import os
import os.path
import errno
import numpy as np
import imageio
import sys
import torch
import torch.utils.data as data
import sparseconvnet as scn

class NeuroMorpho(data.Dataset):
    """ Neuromorpho Dataset.
    Args:
        root (string): Root directory of dataset.
        folder : dataset subfolder
        sample_ids: 1-d array of sample id's
        labels: 1-d array of integer class labels
        img_size: width of sqaure images
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, folder, sample_ids, labels, img_size = 256,
                 transform=None, target_transform=None, rgb=False):

        assert(len(sample_ids) == len(labels))
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        for i, nid in enumerate(sample_ids):
            img_path = os.path.join(root, folder, '%d.png' % nid)
            img = imageio.imread(img_path)
            self.data.append(img)
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape(len(sample_ids), img_size, img_size)
        if rgb:
            self.data = np.stack([self.data]*3, axis=3)
        self.labels = labels
        # now load the picked numpy arrays

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def SparseMerge(spatial_size=255):
    def merge(tbl):
        targets=[x[1] for x in tbl]
        locations=[]
        features=[]
        for idx, (img, tar) in enumerate(tbl):
            #print(np.array(img).shape) (256, 256)
            coords = np.nonzero(np.array(img)[0:spatial_size, 0:spatial_size])
            coords = torch.LongTensor(np.vstack(coords).T)
            coords = torch.cat([coords.long(), torch.LongTensor([idx]).expand([coords.size(0), 1])], 1)
            locations.append(coords)
            f = torch.ones([coords.size(0), 1])
            f = torch.cat([f, torch.ones([coords.size(0), 1])], 1)
            features.append(f)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)),
                'target': torch.LongTensor(targets)}
    return merge


class NeuroMorphoSparse3D(NeuroMorpho):
    """ Neuromorpho Dataset.
    Args:
        root (string): Root directory of dataset.
        folder : dataset subfolder
        sample_ids: 1-d array of sample id's
        labels: 1-d array of integer class labels
        img_size: width of sqaure images
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    pass
