import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import util.util as util


class alignedDataset(BaseDataset):
    """
    This dataset has been editted by minseok for paird setting.

    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size, 'A'))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size, 'B'))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        transform_list = []
        transform_list.append(transforms.Resize((256,256)))
        transform_list.append(transforms.ColorJitter(
            brightness=0.2,    # Adjust brightness by a factor of 0.8 to 1.2
            contrast=0.2,      # Adjust contrast by a factor of 0.8 to 1.2
            saturation=0.2,    # Adjust saturation by a factor of 0.8 to 1.2
            hue=0.1            # Adjust hue by a factor of -0.1 to 0.1
        ))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

        self.transfrom = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        if self.opt.input_nc != 1:
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
        else:
            A_img = Image.open(A_path).convert('L')
            B_img = Image.open(B_path).convert('L')

        params = {
            'crop_pos': (random.randint(0, A_img.size[0] - self.opt.crop_size), random.randint(0, A_img.size[1] - self.opt.crop_size)),
            'flip': random.random() > 0.5,
            'scale_factor': random.uniform(0.8, 1.2)  # Example scale factor for zooming
        }


        A = self.transfrom(A_img)
        B = self.transfrom(B_img)

        if random.random() > 0.5:
            A = torch.flip(A, dims=[2])  # Horizontal flip
            B = torch.flip(B, dims=[2])
        if random.random() > 0.5:
            A = torch.flip(A, dims=[1])  # Vertical flip
            B = torch.flip(B, dims=[1])
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
