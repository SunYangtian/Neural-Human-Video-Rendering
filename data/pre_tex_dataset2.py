######  load both pose image (9 channels) and Laplace feature (24x3 channels) as input
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2

def tex_im2tensor(tex_im, tex_size):
    '''
    change texture image [3,h,w] to tensor [part_numx3, tex_size, tex_size] 
    '''
    tex_tensor = torch.zeros([24,3,tex_size,tex_size]) # [part_num, 3, tex_size, tex_size]
    for i in range(4):
        for j in range(6):
            tex_tensor[(6*i+j),:,:,:] = tex_im[:, (tex_size*j):(tex_size*j+tex_size),
                                                    (tex_size*i):(tex_size*i+tex_size)]
    tex_tensor = torch.flip(tex_tensor, dims=[2]) # do vertical flip
    tex_tensor = tex_tensor.contiguous().view(-1, tex_size, tex_size) # [part_num x 3, tex_size, tex_size]
    return tex_tensor

def mask_im2tensor(mask_im, tex_size):
    '''
    change mask image [1,h,w] to tensor [part_num, tex_size, tex_size] 
    '''
    mask_tensor = torch.zeros([24,1,tex_size,tex_size]) # [part_num, 3, tex_size, tex_size]
    for i in range(4):
        for j in range(6):
            mask_tensor[(6*i+j),:,:,:] = mask_im[:, (tex_size*j):(tex_size*j+tex_size),
                                                    (tex_size*i):(tex_size*i+tex_size)]
    mask_tensor = torch.flip(mask_tensor, dims=[2]) # do vertical flip
    mask_tensor = mask_tensor.contiguous().view(-1, tex_size, tex_size) # [part_num x 3, tex_size, tex_size]
    return mask_tensor

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (pose images)
        self.dir_A = opt.pose_path
        self.A_paths = sorted(make_dataset(self.dir_A))
        if opt.use_laplace:
            self.dir_A_Lap = os.path.join(opt.lapalce_path, '123')
            self.dir_A2_Lap = os.path.join(opt.lapalce_path, '456')
            self.A_Lap_paths = sorted(make_dataset(self.dir_A_Lap))
            self.A2_Lap_paths = sorted(make_dataset(self.dir_A2_Lap))

        self.dir_A_tex = opt.pose_texture_path
        self.A_tex_paths = sorted(make_dataset(self.dir_A_tex))

        ### set transform
        A = Image.open(self.A_paths[0]).convert('RGB')
        params = get_params(self.opt, A.size, A.mode)
        self.transform = get_transform(self.opt, params)

        ### set tex transform
        tex_size = self.tex_size = opt.tex_size
        texture = Image.open(opt.texture_path).convert('RGB')
        if texture.width // 4 != self.tex_size:
            texture = texture.resize((self.tex_size*4, self.tex_size*6))
        tex_transform = [transforms.ToTensor()] # set to [0,1]
        # , transforms.Normalize((0.5,)*3, (0.5,)*3) # set to [-1,1]
        self.tex_transform = transforms.Compose(tex_transform)
        texture = self.tex_transform(texture) # [3, h, w]

        ### partial(dynamic) texture image
        self.dir_tex = opt.part_texture_path
        self.tex_paths = sorted(make_dataset(self.dir_tex))

        ### texture image and transform
        tex_transform = [transforms.ToTensor()] # set to [0,1]
        # , transforms.Normalize((0.5,)*3, (0.5,)*3) # set to [-1,1]
        self.tex_transform = transforms.Compose(tex_transform)
        self.tex_size = opt.tex_size

        ### initial texture
        initial_tex = Image.open(opt.texture_path).convert('RGB') # (width,height)
        if initial_tex.width != self.tex_size * 4:
            initial_tex = initial_tex.resize((self.tex_size*4,self.tex_size*6))
        initial_tex = self.tex_transform(initial_tex)
        self.initial_tex = tex_im2tensor(initial_tex, self.tex_size)
        
        self.dataset_size = len(self.A_paths)
        assert(len(self.tex_paths) == self.dataset_size), \
            "nums of dynamic texture (%d) is not equal to pose image (%d)" % (len(self.tex_paths), self.dataset_size)
        assert(len(self.A_tex_paths) == self.dataset_size), \
            "nums of laplace texture (%d) is not equal to pose image (%d)" % (len(self.A_tex_paths), self.dataset_size)
        # assert (opt.loadSize == self.tex_size), "image size (%d) is not equal to tex size (%d)" % (opt.loadSize, self.tex_size)

    def __getitem__(self, index):
        index = index % self.dataset_size

        ### input A (pose images)
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        A_tensor = self.transform(A)
        if self.opt.use_laplace:
            A_Lap_path = self.A_Lap_paths[index]
            A_Lap = Image.open(A_Lap_path).convert('RGB')
            A_Lap_tensor = self.transform(A_Lap)
            A2_Lap_path = self.A2_Lap_paths[index]
            A2_Lap = Image.open(A2_Lap_path).convert('RGB')
            A2_Lap_tensor = self.transform(A2_Lap)
            
            A_tensor = torch.cat([A_tensor, A_Lap_tensor, A2_Lap_tensor], dim=0)

        a_tensor = torch.nn.functional.upsample(A_tensor.unsqueeze(0), size=self.tex_size, mode='bilinear')
        a_tensor = a_tensor[0]

        A_tex_path = self.A_tex_paths[index]
        A_tex = Image.open(A_tex_path).convert('RGB')
        if A_tex.width != self.tex_size * 4:
            A_tex = A_tex.resize((self.tex_size*4, self.tex_size*6))
        A_tex_tensor = self.tex_transform(A_tex)
        A_tex_tensor = tex_im2tensor(A_tex_tensor, self.tex_size)

        a_tensor = torch.cat([a_tensor, A_tex_tensor], dim=0)

        ### partial texture
        tex_path = self.tex_paths[index]
        texture_img = Image.open(tex_path).convert('RGB')
        if texture_img.width != self.tex_size * 4:
            texture_img = texture_img.resize((self.tex_size*4,self.tex_size*6))
        texture_img = self.tex_transform(texture_img) # [3, h, w]
        tex_tensor = tex_im2tensor(texture_img, self.tex_size) # [24x3, tex_size, tex_size]

        mask_img = torch.zeros((self.tex_size*6, self.tex_size*4), dtype=torch.float32) # [1, h, w]
        mask_img[torch.sum(texture_img, dim=0)>0] = 1
        mask_tensor = mask_im2tensor(mask_img.unsqueeze(0), self.tex_size) # [24x3, tex_size, tex_size]

        input_dict = {'initial_tex': self.initial_tex, 'part_tex': tex_tensor, 'pose': a_tensor, 'mask': mask_tensor}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Pre_Texture_Dataset (pose+laplace as input)'

if __name__ == "__main__":
    pass