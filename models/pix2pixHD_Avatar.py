import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
import cv2, copy

from PIL import Image
import torchvision.transforms as transforms
def read_texture_to_tensor(texture_path, tex_size):
    texture = Image.open(texture_path).convert('RGB')
    if texture.width // 4 != tex_size:
        texture = texture.resize((tex_size*4, tex_size*6))
    tex_transform = transforms.Compose([transforms.ToTensor()]) # set to [0,1]
    texture_tensor = tex_transform(texture) # [3, h, w]
    return texture_tensor

def read_bg_to_tensor(bg_path, bg_size):
    bg = Image.open(bg_path).convert('RGB')
    bg_transform = [transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)] # set to [-1,1]
    bg_transform = transforms.Compose(bg_transform)
    bg = bg_transform(bg) # [3, bg_h, bg_w]
    bg = torch.nn.functional.upsample(bg.unsqueeze(0), size=bg_size, mode='bilinear') # [3, h, w]
    return bg

import copy
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
    return tex_tensor.unsqueeze(0)

def print_grad(grad):
    # grad = grad * 10000
    print("Gradient:", grad.max(), ".............")

def split_consecutive_tensor(x):
    '''
        assume x has size [bs*2, c, h, w]
    '''
    bs = x.shape[0] // 2
    return x[:bs], x[bs:]

class Pix2PixHD_Avatar(BaseModel):
    def name(self):
        return 'Pix2PixHD_Avatar'
    
    def init_loss_filter(self):
        if self.opt.use_densepose_loss:
            flags = (True, True, True, True, True, True, True, True, True, True)
        else:
            flags = (True, True, True, True, True, True, False, False, True, True)
        def loss_filter(g_gan, L2, mask, g_vgg, d_real, d_fake, uv, prob, mask_tex, temp):
            return [l for (l,f) in zip((g_gan,L2,mask,g_vgg,d_real,d_fake, uv, prob, mask_tex, temp),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc # used for TexG and TransG

        ##### define networks
        # Generator network
        TexG_input_nc = 72 + opt.num_class*3
        if opt.pose_plus_laplace:
            TexG_input_nc = 81 + opt.num_class*3
            TexG_input_nc = 81 

        # initialize device information
        self.device = 'cuda:{}'.format(self.gpu_ids[0])

        if opt.isTrain:
            texture_im = read_texture_to_tensor(opt.texture_path, opt.tex_size)
            # (24x3, tex_size, tex_size)
            texture_tensor = tex_im2tensor(texture_im, opt.tex_size) # (1, 24x3, tex_size, tex_size)
            tmp_tensor = torch.zeros_like(texture_tensor).repeat(1,5,1,1) # (1, 24x3x5, tex_size, tex_size)
            texture_tensor = util.catTextureTensor(texture_tensor, tmp_tensor)
            self.texture = torch.nn.Parameter(texture_tensor.to(device=self.device)) 
        # self.texture.register_hook(print_grad)

        TransG_input_nc = input_nc
        self.TransG = networks.define_G(TransG_input_nc, (opt.num_class*2,opt.num_class+1), opt.ngf_translate, opt.TransG,
                                      opt.n_downsample_translate, opt.n_blocks_translate, opt.n_local_enhancers,
                                      opt.norm, gpu_ids=self.gpu_ids)

        self.Feature2RGB = networks.FgFeature2RGB().cuda()

        if opt.isTrain:
            bg_tensor = read_bg_to_tensor(opt.bg_path, opt.loadSize)
            self.BG = torch.nn.Parameter(bg_tensor.to(device=self.device)) # (3,h,w)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + 3 # pose + generated image
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # load networks
        TEST = not self.isTrain
        if TEST:
            pretrained_path = os.path.join(opt.checkpoints_dir,opt.name)
            texture_path = os.path.join(pretrained_path, "%s_texture.npy" % opt.which_epoch)
            texture_tensor = torch.tensor(np.load(texture_path))
            self.texture = torch.nn.Parameter(texture_tensor.to(device=self.device))

            bg_path = os.path.join(pretrained_path, "%s_bg.jpg" % opt.which_epoch)
            # bg_path = "/apdcephfs/share_1364276/alyssatan/checkpoints/dance15_18Feature_Temporal/fire_new.jpg"
            bg_tensor = read_bg_to_tensor(bg_path, opt.loadSize)
            self.BG = torch.nn.Parameter(bg_tensor.to(device=self.device)) # (3,h,w)

            self.load_network(self.TransG, 'TransG', opt.which_epoch, pretrained_path)
            self.load_network(self.Feature2RGB, 'Feature2RGB', opt.which_epoch, pretrained_path)

        elif not self.isTrain or opt.continue_train:
            pretrained_path = os.path.join(opt.checkpoints_dir,opt.name)

            texture_path = os.path.join(opt.load_pretrain_TransG, "%s_texture.jpg" % opt.which_epoch_TransG)
            texture_im = read_texture_to_tensor(texture_path, opt.tex_size)
            texture_tensor = tex_im2tensor(texture_im, opt.tex_size)
            tmp_tensor = torch.zeros_like(texture_tensor).repeat(1,5,1,1) # (1, 24x3x5, tex_size, tex_size)
            texture_tensor = util.catTextureTensor(texture_tensor, tmp_tensor)
            self.texture = torch.nn.Parameter(texture_tensor.to(device=self.device))

            bg_path = os.path.join(opt.load_pretrain_TransG, "%s_bg.jpg" % opt.which_epoch_TransG)
            bg_tensor = read_bg_to_tensor(bg_path, opt.loadSize)
            self.BG = torch.nn.Parameter(bg_tensor.to(device=self.device)) # (3,h,w)

            self.load_network(self.TransG, 'TransG', opt.which_epoch_TransG, opt.load_pretrain_TransG)

            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch_TransG, opt.load_pretrain_TransG)      

        # has been pre-trained
        else:
            self.load_network(self.TransG, 'TransG', opt.which_epoch_TransG, opt.load_pretrain_TransG)

        if self.opt.verbose or True:
            print('---------- Networks initialized -------------')
            print('parameter number of TransG: %s' % sum(p.numel() for p in self.TransG.parameters()))
            # print('parameter number of TexG: %s' % sum(p.numel() for p in self.TexG.parameters()))
            print('parameter number of Feature2RGB: %s' % sum(p.numel() for p in self.Feature2RGB.parameters()))
            print('parameter number of texture: %s' % sum(p.numel() for p in self.texture))
            if self.isTrain:
                print('parameter number of netD: %s' % sum(p.numel() for p in self.netD.parameters()))

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr2

            # define loss functions
            self.loss_filter = self.init_loss_filter()
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor) 
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1_mask = torch.nn.L1Loss(reduction='none')
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL2_mask = torch.nn.MSELoss(reduction='none')
            self.criterionMask = torch.nn.BCEWithLogitsLoss()
            # self.criterionMask = torch.nn.BCELoss()
            self.criterion_UV = torch.nn.L1Loss(reduction='none')
            self.criterion_Prob = torch.nn.CrossEntropyLoss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','L2', 'mask','G_VGG','D_real', 'D_fake', \
                                                'UV_loss', 'Probs_loss', 'mask_human', 'temporal')

            meshgrid = torch.stack(torch.meshgrid(torch.linspace(-1,1,200), torch.linspace(-1,1,200)), dim=-1).to(self.texture.device)
            # self.embedder, _ = networks.get_embedder(5)
            # self.meshgrid = self.embedder(meshgrid).unsqueeze(0).permute(0,3,1,2)

            # optimizer D
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.lr = opt.lr
            self.beta1 = opt.beta1
            # self.FIRST_EPOCH = 13 # for the small amount data
            self.FIRST_EPOCH = 1 # for All data
            if opt.continue_train:
                self.FIRST_EPOCH = 1 # for pretrained

            # save tenGrid for given size image
            self.backwarp_tenGrid = {}

        # initialize device information
        if len(self.gpu_ids) > 0:
            self.device = 'cuda:{}'.format(self.gpu_ids[0])
        else:
            self.device = 'cpu'

    def init_optimizer_G(self, epoch, StaticEpoch=3):
        if epoch <= StaticEpoch:
            print("The epoch is %d, Static update !" % epoch)
            ratio = 0.9 ** (epoch)
            self.optimizer_G = torch.optim.Adam([{'params': self.TransG.parameters(), 'lr': self.lr*ratio},
                                                 {'params': self.texture, 'lr': self.lr*ratio},
                                                 {'params': self.Feature2RGB.parameters(), 'lr': self.lr*ratio},
                                                 {'params': self.BG, 'lr': self.lr*ratio}], lr=self.lr, betas=(self.beta1, 0.999))
        else:
            ratio = 0.9 ** (epoch)
            print("The epoch is %d, decrease TransG !" % epoch)
            # update All
            self.optimizer_G = torch.optim.Adam([{'params': self.TransG.parameters(), 'lr': self.lr*ratio},
                                                 {'params': self.texture, 'lr': self.lr},
                                                 {'params': self.Feature2RGB.parameters(), 'lr': self.lr},
                                                 {'params': self.BG, 'lr': self.lr}], lr=self.lr, betas=(self.beta1, 0.999))
        return self.optimizer_G

    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.size()) not in self.backwarp_tenGrid:
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            tenGrid = torch.cat([tenHorizontal, tenVertical], dim=1)
            self.backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical], 1).to(self.device)
        # end
        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
        return torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

    def cal_conf(self, flow, flow2, Conf_Thresh=1):
        '''
        calculating the confidence (0/1) of each value in flow
        '''
        grid = util.tensorGrid(flow).to(flow.device)
        B, _, H, W = flow.size()
        conf = torch.zeros_like(flow[:,:1,:,:])
        F1to2 = grid + flow # each grid represent point coord in original image, value is new coord in new image [B,2,H,W]
        F1to2[:,0,:,:] = F1to2[:,0,:,:]*2/W - 1
        F1to2[:,1,:,:] = F1to2[:,1,:,:]*2/H - 1

        B2to1 = torch.nn.functional.grid_sample(flow2, F1to2.permute(0,2,3,1), mode='bilinear', padding_mode='border') # diff of new coord point to original point
        # F1to2 is point coord, B2to1 is flow (diff)
        conf[torch.pow((flow + B2to1),2).sum(dim=1, keepdim=True) < Conf_Thresh] = 1
        return conf

    # only for single tensor or list of tensors 
    def encode_input(self, origin):
        if isinstance(origin, list):
            encoded = []
            for item in origin:
                encoded.append(self.encode_input(item))
        else:
            encoded = Variable(origin.data.cuda())
        return encoded

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    # load from tensor [bs, 3, h, w], only for bs = 1
    def pre_load_texture(self):
        num_f = self.tex_face.shape[0]
        tex_res = self.tex_res
        wxy = torch.arange(num_f*tex_res*tex_res, device=self.device).repeat(2,1).transpose(1,0).to(torch.int32) # num_f*res*res, 2 dtype=int
        wxy[:,0] = wxy[:,0] % tex_res # wx
        wxy[:,1] = wxy[:,1] % (tex_res*tex_res) // tex_res # wy

        wxy_f = wxy.to(torch.float32)
        weight = torch.zeros(num_f*tex_res*tex_res, 2, device=self.device) # dtype=float32
        weight[torch.sum(wxy,1)<tex_res] = (wxy_f[torch.sum(wxy,1)< tex_res]+1/3) / tex_res
        weight[torch.sum(wxy,1)>=tex_res] = (tex_res-1-wxy_f[torch.sum(wxy,1)>= tex_res]+2/3) / tex_res
        weight_ = torch.unsqueeze(1 - torch.sum(weight, dim=1), dim=1)
        weight = torch.cat((weight, weight_), dim=1) # num_f*res*res, 3
        weight = torch.unsqueeze(weight, dim=1) # num_f*res*res, 1, 3

        sample_face = self.tex_face.unsqueeze(1).repeat(1,tex_res*tex_res,1,1).view(-1,3,2) # num_f*res*res, 3, 2
        sample_pos = torch.bmm(weight, sample_face) # num_f*res*res, 1, 2
        grid = sample_pos.unsqueeze(0)*2 - 1 # [1, num_f*res*res, 1, 2]
        return grid

    def load_texture(self, tex_img):
        texture = torch.nn.functional.grid_sample(tex_img, self.grid).permute(0,2,3,1) # [1, 3, num_f*res*res, 1]
        return texture.reshape(-1, self.tex_res**2, 3)

    # def forward(self, epoch, texture, Pose, pose, mask, real, real_uv, real_cls, bg, infer=False):
    def forward(self, epoch, texture, Pose, mask, real, real_uv, real_cls, bg, \
        Pose_before, mask_before, real_image_before, real_uv_before, real_cls_before, flow, flow_inv):

        # Encode Inputs
        texture = self.encode_input([texture])
        Pose, mask, real_image = self.encode_input([Pose, mask, real])
        real_uv, real_cls = self.encode_input([real_uv, real_cls])
        bg = self.encode_input(bg)

        Pose_before, mask_before, real_image_before, real_uv_before, real_cls_before = self.encode_input([\
            Pose_before, mask_before, real_image_before, real_uv_before, real_cls_before])
        flow, flow_inv = self.encode_input([flow, flow_inv])

        # merge conseuctive image
        real_images = torch.cat([real_image, real_image_before], dim=0)
        real_uvs = torch.cat([real_uv, real_uv_before], dim=0)
        real_clss = torch.cat([real_cls, real_cls_before], dim=0)
        masks = torch.cat([mask, mask_before], dim=0)

        mask_tex = 0

        TransG_input = torch.cat([Pose, Pose_before], dim=0)
        UVs, Probs = self.TransG(TransG_input)

        gen_texture = self.texture

        fg_image = util.texture2image(gen_texture, UVs, Probs)
        fg_image_raw = util.texture2image(gen_texture, UVs, Probs, selNUM=3)
        fg_image_raw = fg_image_raw*2 - 1

        fg_image = self.Feature2RGB(fg_image)

        bg_image = self.BG

        # new bg generation
        norm_Probs = torch.nn.functional.softmax(Probs, dim=1)
        bg_mask = norm_Probs[:,0:1,:,:]
        # bg_mask = 1 - feature_mask # change the mask to the output of FgFeatureRGB
        fake_image = fg_image * (1-bg_mask) + bg_image * (bg_mask)
        fake_image_raw = fg_image_raw * (1-bg_mask) + bg_image * (bg_mask)

        StaticEpoch = 3
        self.StaticEpoch = StaticEpoch

    ### GAN loss ###
        # Fake Detection and Loss
        # pred_fake_pool_raw = self.discriminate(TransG_input, fake_image_raw, use_pool=False)
        # loss_D_fake = self.criterionGAN(pred_fake_pool_raw, False)
        loss_D_fake = torch.tensor(0, dtype=torch.float, requires_grad=True).to(self.device)*0
        if epoch > StaticEpoch:
            pred_fake_pool = self.discriminate(TransG_input, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            # loss_D_fake *= 0.5

        # Real Detection and Loss
        loss_D_real = torch.tensor(0, dtype=torch.float, requires_grad=True).to(self.device)*0
        if epoch > StaticEpoch:
            pred_real = self.discriminate(TransG_input, real_images)
            loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        # pred_fake_raw = self.netD.forward(torch.cat((TransG_input, fake_image_raw), dim=1))    
        # loss_G_GAN = self.criterionGAN(pred_fake_raw, True)
        loss_G_GAN = 0
        if epoch > StaticEpoch:
            pred_fake = self.netD.forward(torch.cat((TransG_input, fake_image), dim=1))    
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            # loss_G_GAN *= 0.5

    ### VGG loss ###
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG += self.criterionVGG(fake_image_raw.contiguous(), real_images.contiguous()) * self.opt.lambda_feat
            if epoch > StaticEpoch:
                loss_G_VGG += self.criterionVGG(fake_image.contiguous(), real_images.contiguous()) * self.opt.lambda_feat
                # loss_G_VGG *= 0.5

    ### L2 loss ###
        loss_L2_mask = 0
        loss_L2_mask += self.criterionL2_mask(fake_image_raw, real_images) * self.opt.lambda_L2
        if epoch > StaticEpoch:
            loss_L2_mask += self.criterionL2_mask(fake_image, real_images) * self.opt.lambda_L2
            # loss_L2_mask *= 0.5
        loss_L2_mask = torch.mean(loss_L2_mask)
    
    ### UV loss ###
        loss_UV = 0
        UV_mask=[]
        for part_id in range(24):
            UV_mask.append(real_clss==(part_id+1))
            UV_mask.append(real_clss==(part_id+1))
        UV_mask_tensor=torch.stack(UV_mask,dim=1).float()
        loss_UV = self.criterion_UV(UVs, real_uvs) * UV_mask_tensor * self.opt.lambda_UV
    ### classify loss ###
        loss_Prob = self.criterion_Prob(Probs, real_clss.long()) * self.opt.lambda_Prob
    ### mask loss ###
        loss_G_mask = 0
         # foreground mask (mask is set to [0,1])
        # loss_mask = self.criterionMask(1-bg_mask, mask) * self.opt.lambda_mask
        loss_mask = self.criterionMask(1-bg_mask, masks) * self.opt.lambda_mask/10
        # loss_mask = self.criterionMask(1-bg_mask, torch.ones_like(mask)) * self.opt.lambda_mask/10 # for dance29 debug
    ### TV loss ###
        # loss_TV = util.All_TVloss(UVs, Probs)
        loss_TV = 0

    ### mask loss ###
        loss_mask_human = 0
        if epoch > StaticEpoch:
            # loss_mask_human = self.criterionL2_mask(fg_image, -1*torch.ones_like(fg_image)) * bg_mask * self.opt.lambda_L2
            loss_mask_human += self.criterionL2_mask(fake_image, real_images) * (1-bg_mask) * self.opt.lambda_L2
            # loss_mask_human *= 0.5
            loss_mask_human = torch.mean(loss_mask_human)

    ### Temporal loss ###
        loss_T = 0
        fake_image_bf = warped_image = warped_real_image = warped_image_comp = conf = torch.zeros_like(fake_image)
        if epoch > StaticEpoch:
            conf = self.cal_conf(flow_inv, flow)
            fake_image_now, fake_image_bf = split_consecutive_tensor(fake_image)
            warped_image = self.backwarp(fake_image_bf, flow_inv)
            warped_real_image = self.backwarp(real_image_before, flow_inv)
            warped_image_comp = warped_image * conf
            loss_T = self.criterionL1_mask(warped_image, fake_image_now) * conf * self.opt.lambda_Temp
            loss_T = torch.mean(loss_T)

#         if self.opt.use_mask_tex_loss:
#             loss_mask_tex = self.criterionMask(mask_tex, real_mask_tex) * self.opt.lambda_mask_tex
        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_L2_mask, loss_mask, loss_G_VGG, loss_D_real, loss_D_fake, loss_UV, loss_Prob, loss_mask_human, loss_T), \
            fg_image, fg_image_raw, bg_image, bg_mask, fake_image, util.pickbaseTexture(gen_texture), UVs, Probs, mask_tex, \
                fake_image_bf, warped_image, warped_real_image, warped_image_comp, conf]

    def inference(self, texture, Pose, bg):
        # Encode Inputs
        # texture, Pose, pose, bg = self.encode_input([texture, Pose, pose, bg])
        texture, Pose, bg = self.encode_input([texture, Pose, bg])

        with torch.no_grad():
            mask_tex = 0
            gen_texture = self.texture
            UVs, Probs = self.TransG(Pose) # [bs,48,h,w], [bs,25,h,w]

            fg_image = util.texture2image(gen_texture, UVs, Probs) # [bs, feature, h, w]

            fg_image_raw = util.texture2image(gen_texture, UVs, Probs, selNUM=3)
            fg_image_raw = fg_image_raw*2 - 1

            fg_image = self.Feature2RGB(fg_image)

            bg_image = self.BG
            Probs = torch.nn.functional.softmax(Probs, dim=1)
            bg_mask = Probs[:,0:1,:,:]

            fake_image = fg_image * (1-bg_mask) + bg_image * (bg_mask)
            fake_image_raw = fg_image_raw * (1-bg_mask) + bg_image * (bg_mask)

        # return fg_image, bg_image, bg_mask, fg_image_raw*(1-bg_mask) + (1)*(bg_mask), gen_texture, UVs, Probs, mask_tex
        return fg_image, bg_image, bg_mask, fake_image, gen_texture, UVs, Probs, mask_tex
        # return fg_image, bg_image, bg_mask, fake_image_raw, gen_texture, UVs, Probs, mask_tex
        # return fg_image, bg_image, bg_mask, fg_image_raw, gen_texture, UVs, Probs, mask_tex


    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                     
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        # self.save_network(self.TexG, 'TexG', which_epoch, self.gpu_ids)
        self.save_network(self.TransG, 'TransG', which_epoch, self.gpu_ids)
        self.save_network(self.Feature2RGB, 'Feature2RGB', which_epoch, self.gpu_ids)
        # self.save_network(self.BGnet, 'BGnet', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the TransG generator for a number of iterations, also start finetuning it
        params = list(self.TexG.parameters())
        params += list(self.TransG.parameters())
        params += list(self.BGnet.parameters()) 
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_all_params(self):
        params = list(self.TransG.parameters())
        params += list(self.BGnet.parameters())
        params += [self.texture]
        print("learning rate : %s ... " % self.opt.lr)
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def update_texture_params(self):
        # update TransG and BGnet parameters 
        params = [self.texture]
        params += list(self.BGnet.parameters())
        lr = self.opt.lr * 10
        print("learning rate : %s ... " % lr)
        self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(self.opt.beta1, 0.999))

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHD_Avatar):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

def TVloss(x, Probs, idx=1):
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    # x with the shape [bs, c, h, w]
    TVLoss_weight = 1
    batch_size = x.size()[0]
    c_x = x.size()[1]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])

    PART_NUM = Probs.shape[1]
    Probs = torch.nn.functional.softmax(Probs, dim=1)
    # Probs = torch.nn.functional.softmax(Probs.view(batch_size, PART_NUM, -1), dim=-1).view(batch_size, PART_NUM, h_x, w_x)
    prob = Probs[:,idx,:,:].unsqueeze(1) # [bs,1,h,w]
    prob = torch.nn.functional.softmax(prob.view(batch_size, 1, -1), dim=-1).view(batch_size, 1, h_x, w_x)
    print(prob.max(), prob.min())
    cv2.imwrite('./test_prob.jpg', prob[0].permute(1,2,0).detach().cpu().numpy()*255*h_x*w_x/2)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)
    h_tv = (h_tv * prob[:,:,:-1,:]).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)
    w_tv = (w_tv * prob[:,:,:,:-1]).sum()
    return TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size