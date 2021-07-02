import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .inpaintor import InpaintSANet
from collections import OrderedDict
import os

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def my_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG_name, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[], my_init=False, add_mask=False):
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG_name == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG_name == 'gauss':    
        netG = GaussGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG_name == 'part':
        netG = PartGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, add_mask=add_mask)    
    elif netG_name == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG_name == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG_name == 'multihead':
        assert(len(output_nc) == 2) # "only support 2 branch output"
        out_nc_1, out_nc_2 = output_nc # first is UV, secode is Prob
        netG = MultiHeadGeneraotr(input_nc, out_nc_1, out_nc_2, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG_name == 'warp':
        netG = WarpNet2()
    elif netG_name == 'pose_embedding':
        netG = PoseEmbdding()
    elif netG_name == 'unet':
        netG = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=False)
    else:
        raise('generator not implemented!')
    # print(netG)
    print("%s generator initializes with %d input channel, %s output channel and %d ngf.\n \
            And with %d down/up sample and %d resnet block. Norm layer is %s \n\n" \
            % (netG_name, input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm))
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    else:
        netG.cuda()
    if my_init:
        netG.apply(my_weights_init)
    else:
        netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat) 
    # print(netD)
    print("Multiscale discriminator initialized with %d net and %d layers. And with %d ndf. Use lsgan? %s!"
             % (num_D, n_layers_D, ndf, use_sigmoid))
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]   

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()     
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)

### define generator with gauss noise
class GaussGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GaussGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        model2 = [] # seperate downsample and resnet/upsample
        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model2 += [ResnetBlock(2 * ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample
        # model2 += [nn.ConvTranspose2d(2 * ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                norm_layer(ngf * mult), activation]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model2 += [nn.ConvTranspose2d(2 * ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(ngf * mult), activation]
        model2 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(*model2)

    def forward(self, input): # [bs, 3, 512, 512]
        c = self.model(input) # [bs, ngf*(2**n_downsampling), 512/(2**n_downsampling), 512/(2**n_downsampling)]
        z = torch.randn_like(c)
        # cat condition and noise in feature dimension
        x = torch.cat([z,c], dim=1) # [bs, 2*ngf*(2**n_downsampling), 512/(2**n_downsampling), 512/(2**n_downsampling)]
        return self.model2(x)

### define generator with gauss noise
class PartGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', add_mask=False):
        assert(n_blocks >= 0)
        super(PartGenerator, self).__init__()     
        activation = nn.ReLU(True)
        self.add_mask = add_mask    

        # add layer in 0810
        model_enc = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf*2, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_enc += [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model_enc += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks // 2):
            model_enc += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.model_enc = nn.Sequential(*model_enc)

        ### upsample
        self.partNUM = output_nc // 3
        print("initialize total %d part decoders ... " % self.partNUM)
        for partID in range(self.partNUM):
            model_dec = []
            mult = 2**n_downsampling
            for i in range(n_blocks // 2):
                model_dec += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model_dec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
            setattr(self, "decoder_"+str(partID), nn.Sequential(*model_dec))

            model_out = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]        
            setattr(self, "out_"+str(partID), nn.Sequential(*model_out))

            if add_mask:
                model_out_mask = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]        
                setattr(self, "mask_"+str(partID), nn.Sequential(*model_out_mask))

            
    def forward(self, input):
        latent = self.model_enc(input)
        output = []
        output_mask = []

        if self.add_mask:
            for partID in range(self.partNUM):
                model_dec = getattr(self, "decoder_"+str(partID)) # get decoder model
                decoded = model_dec(latent) # get decoded feature
                model_out = getattr(self, "out_"+str(partID)) # get out model
                part_out = model_out(decoded) # get residual 
                model_mask = getattr(self, "mask_"+str(partID)) # get mask model
                mask = model_mask(decoded) # get mask
                output.append(part_out)
                output_mask.append(mask) # set to (0,1)
            return torch.cat(output, dim=1), torch.cat(output_mask, dim=1)
        else:
            for partID in range(self.partNUM):
                model_dec = getattr(self, "decoder_"+str(partID))
                decoded = model_dec(latent)
                model_out = getattr(self, "out_"+str(partID))
                part_out = model_out(decoded)
                output.append(part_out)
            return torch.cat(output, dim=1)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input):
        outputs = self.model(input)                   
        return outputs

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # vgg_model_path = '/apdcephfs/share_1330077/yangtiansun/pix2pixHD_avatar_08/checkpoints/vgg19-dcbb9e9d.pth'
        vgg_model_path = '../104mnt/DanceDataset/vgg19-dcbb9e9d.pth'
        if os.path.exists(vgg_model_path):
            Vgg19_model = models.vgg19()
            Vgg19_model.load_state_dict(torch.load(vgg_model_path))
            vgg_pretrained_features = Vgg19_model.features
        else:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

### Added ###
class MultiHeadGeneraotr(nn.Module):
    def __init__(self, input_nc, output_nc_1, output_nc_2, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        super(MultiHeadGeneraotr, self).__init__()
        activation = nn.ReLU(True)        

        # model_d = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            # model_d += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                    #   norm_layer(ngf * mult * 2), activation]
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        model_r = []
        for i in range(n_blocks):
            # model_r += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        model_u = []         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model_u += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                    #    norm_layer(int(ngf * mult / 2)), activation]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]

        # Final layers (may be several per branch)
        final_layer1 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc_1, kernel_size=7, padding=0), nn.Tanh()]
        final_layer2 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc_2, kernel_size=7, padding=0)]

        # self.model_d = nn.Sequential(*model_d)
        # self.model_r = nn.Sequential(*model_r)
        # self.model_u = nn.Sequential(*model_u)
        self.model = nn.Sequential(*model)

        self.final_layer1 = nn.Sequential(*final_layer1) # for UVs
        self.final_layer2 = nn.Sequential(*final_layer2) # for Probs

    def forward(self, input):
        # feature = self.model_d(input) # [4,9,512,512] -> [4,1024,32,32]
        # feature = self.model_r(feature) # [4,1024,32,32]
        # feature = self.model_u(feature)

        feature = self.model(input)

        out_UVs = self.final_layer1(feature)
        out_Probs = self.final_layer2(feature)
        return out_UVs, out_Probs # Probs need to be Softmax!

class DeepFill(object):
    def __init__(self, load_path, gpu_ids):
        super(DeepFill, self).__init__()
        self.bgnet = self.create_bgnet(load_path, gpu_ids)

    def create_bgnet(self, load_path, gpu_ids):
        net = InpaintSANet(c_dim=4)
        self._load_params(net,load_path, need_module=False)
        net.eval()
        net.cuda(gpu_ids[0])
        return net

    def _load_params(self, network, load_path, need_module=False):
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one %s' % load_path

        def load(model, orig_state_dict):
            state_dict = OrderedDict()
            for k, v in orig_state_dict.items():
                # remove 'module'
                name = k[7:] if 'module' in k else k
                state_dict[name] = v

            # load params
            model.load_state_dict(state_dict)

        save_data = torch.load(load_path)
        if need_module:
            network.load_state_dict(save_data)
        else:
            load(network, save_data)

        print('Loading net: %s' % load_path)

##### Define the warp network #####
class WarpNet(nn.Module):
    def __init__(self, input_nc=9, output_nc=48, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(WarpNet, self).__init__()
        activation=nn.ReLU(True)
        encoder = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, stride=2, padding=0), norm_layer(ngf), activation]
        # [bs,9,200,200] -> [bs,32,100,100]
        encoder += [nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), norm_layer(ngf), activation]
        # [bs,32,100,100] -> [bs,32,50,50]
        encoder += [nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1), norm_layer(ngf), activation]
        # [bs,32,50,50] -> [bs,32,25,25]
        self.encoder = nn.Sequential(*encoder)

        self.partNUM = output_nc // 2
        print("initialize total %d part decoders ... " % self.partNUM)
        for partID in range(self.partNUM):
            decoder = []
            decoder += [nn.ConvTranspose2d(ngf, ngf//2, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf//2), activation]
            # [bs,32,25,25] -> [bs,16,50,50]
            decoder += [nn.ConvTranspose2d(ngf//2, ngf//4, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf//4), activation]
            # [bs,16,50,50] -> [bs,8,100,100]
            decoder += [nn.ConvTranspose2d(ngf//4, ngf//8, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf//8), activation]
            # [bs,8,100,100] -> [bs,4,200,200]
            decoder += [nn.ConvTranspose2d(ngf//8, ngf//16, kernel_size=3, stride=1, padding=1), norm_layer(ngf//16), nn.Tanh()]
            # [bs,4,200,200] -> [bs,2,200,200]
            setattr(self, "decoder_"+str(partID), nn.Sequential(*decoder))

    def forward(self, pose):
        '''
            pose: [bs,9,h,w]
            output: [bs,h,w,24*2]
        '''
        bs, _, h, w = pose.shape
        feature = self.encoder(pose)

        outputs = []
        for partID in range(self.partNUM):
            decoder = getattr(self, "decoder_"+str(partID))
            output = decoder(feature)
            outputs.append(output)

        return torch.cat(outputs, dim=1)

##### Define the warp network version2 #####
class WarpNet2(nn.Module):
    def __init__(self, input_nc=3, output_nc=48, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(WarpNet2, self).__init__()
        activation=nn.ReLU(True)
        encoder = [nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=0)), norm_layer(ngf), activation]
        # [bs,3,200,200] -> [bs,8,100,100]
        encoder += [nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1), norm_layer(16), activation]
        # [bs,8,100,100] -> [bs,16,50,50]
        encoder += [nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), norm_layer(32), activation]
        # [bs,16,50,50] -> [bs,32,25,25]
        encoder += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), norm_layer(64), activation]
        # [bs,64,13,13]
        encoder += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), norm_layer(128), activation]
        # [bs,128,7,7]
        encoder += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), norm_layer(256), activation]
        # [bs,256,4,4]
        self.encoder = nn.Sequential(*encoder)

        latent_dim = 256*4*4

        fc = [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]
        fc += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]
        self.fc = nn.Sequential(*fc)

        decoder = []
        decoder += [nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1), norm_layer(128), activation]
        decoder += [nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1), norm_layer(64), activation]
        decoder += [nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=3, stride=2, padding=1), nn.Tanh()]
        # [bs,48,25,25]
        self.decoder = nn.Sequential(*decoder)

        ### init weight ###
        # self.encoder.weight.data.fill_(0)
        # self.fc.weight.data.fill_(0)
        # self.decoder.weight.data.fill_(0)
        self.init_weight(self.encoder)
        self.init_weight(self.fc)
        self.init_weight(self.decoder)

        ygrid, xgrid = np.meshgrid(
                np.linspace(-1.0, 1.0, 200),
                np.linspace(-1.0, 1.0, 200), indexing='ij')
        self.grid = torch.tensor(np.stack((xgrid, ygrid), axis=0)[None].astype(np.float32)) # [1, 2, 200, 200]

    def init_weight(self, net_module):
        for m in net_module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)


    def forward(self, pose):
        '''
            pose: [bs,3,h,w]
            output: [bs,h,w,24*2]
        '''
        bs, _, h, w = pose.shape
        feature = self.encoder(pose)
        feature = feature.view(bs, -1)

        feature = self.fc(feature)

        feature = feature.view(bs,-1,4,4)
        sparse_warp = self.decoder(feature)

        grid = self.grid.clone().permute(0,2,3,1).repeat(bs,1,1,1).to(sparse_warp.device)
        dense_warp = nn.functional.grid_sample(sparse_warp, grid)
        return dense_warp


##### Define the pose embedding network #####
class PoseEmbdding(nn.Module):
    def __init__(self, input_nc=3, norm_layer=nn.InstanceNorm2d):
        super(PoseEmbdding, self).__init__()
        activation=nn.ReLU(True)
        encoder = [nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=0)), norm_layer(8), activation]
        # [bs,3,200,200] -> [bs,8,100,100]
        encoder += [nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1), norm_layer(16), activation]
        # [bs,8,100,100] -> [bs,16,50,50]
        encoder += [nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), norm_layer(32), activation]
        # [bs,16,50,50] -> [bs,32,25,25]
        encoder += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), norm_layer(64), activation]
        # [bs,64,13,13]
        encoder += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), norm_layer(128), activation]
        # [bs,128,7,7]
        encoder += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), norm_layer(256), activation]
        # [bs,256,4,4]
        self.encoder = nn.Sequential(*encoder)

        latent_dim = 256*4*4

        fc = [nn.Linear(in_features=latent_dim, out_features=latent_dim//4), activation]
        fc += [nn.Linear(in_features=latent_dim//4, out_features=latent_dim//16), activation]
        fc += [nn.Linear(in_features=latent_dim//16, out_features=latent_dim//64), activation]
        self.fc = nn.Sequential(*fc)

    def forward(self, pose):
        '''
            pose: [bs,3,200,200]
            output: [bs,64]
        '''
        bs, _, h, w = pose.shape
        feature = self.encoder(pose)
        feature = feature.view(bs, -1)
        feature = self.fc(feature)
        return feature

class Feature2RGB(nn.Module):
    def __init__(self):
        super(Feature2RGB, self).__init__()
        # in_dim = 3+64
        in_dim =72+64+22
        out_dim = 72
        # latent_dim = (in_dim+out_dim)//2
        latent_dim = 256
        norm_layer = nn.InstanceNorm2d
        activation=nn.ReLU(True)
        model = [nn.Linear(in_features=in_dim, out_features=latent_dim), activation]
        model += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]
        model += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]
        model += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]

        model2 = [nn.Linear(in_features=latent_dim+in_dim, out_features=latent_dim), activation]
        model2 += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]
        model2 += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]
        model2 += [nn.Linear(in_features=latent_dim, out_features=latent_dim), activation]

        model2 += [nn.Linear(in_features=latent_dim, out_features=latent_dim//2), activation]
        model2 += [nn.Linear(in_features=latent_dim//2, out_features=out_dim), nn.Tanh()]
        
        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(*model2)

    def forward(self, fg_image):
        bs, c, h, w = fg_image.shape
        # x = fg_image.view(bs, c, -1)
        x = fg_image.permute(0,2,3,1).reshape(-1,c)
        y = self.model(x)
        y = torch.cat([x,y], dim=1)
        y = self.model2(y)
        return y.reshape(bs, h, w, -1).permute(0,3,1,2)

class FgFeature2RGB(nn.Module):
    def __init__(self, ngf=64):
        super(FgFeature2RGB, self).__init__()
        # input_nc =3+64
        # input_nc = 12
        # input_nc = 3
        input_nc = 18
        # input_nc = 18+3
        output_nc = 3
        norm_layer = nn.InstanceNorm2d
        activation = nn.ReLU(True)
        padding_type='reflect'

        ngf = ngf

        downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        downsample += [nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]
        downsample += [nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=3, stride=2, padding=1), norm_layer(ngf*4), activation]
        downsample += [nn.Conv2d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=3, stride=2, padding=1), norm_layer(ngf*8), activation]
        downsample += [nn.Conv2d(in_channels=ngf*8, out_channels=ngf*8, kernel_size=3, stride=2, padding=1), norm_layer(ngf*8), activation]

        resnet = [ResnetBlock(ngf*8, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        resnet += [ResnetBlock(ngf*8, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        upsample = [nn.ConvTranspose2d(ngf*8, int(ngf*8), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf*4), activation]
        upsample += [nn.ConvTranspose2d(ngf*8, int(ngf*4), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf*2), activation]
        upsample += [nn.ConvTranspose2d(ngf*4, int(ngf*2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf*2), activation]
        upsample += [nn.ConvTranspose2d(ngf*2, int(ngf), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf), activation]

        # mask_head = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]
        img_head = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.downsample = nn.Sequential(*downsample)
        self.resnet = nn.Sequential(*resnet)
        self.upsample = nn.Sequential(*upsample)

        # self.mask_head = nn.Sequential(*mask_head)
        self.img_head = nn.Sequential(*img_head)
    
    def forward(self, X):
        out = self.downsample(X)
        out = self.resnet(out)
        out = self.upsample(out)

        # mask = self.mask_head(out)
        img = self.img_head(out)
        # return mask, img
        # return out
        return img

class FgFeature2RGBLocal(nn.Module):
    def __init__(self):
        super(FgFeature2RGBLocal, self).__init__()
            
        self.n_local_enhancers = n_local_enhancers = 1
        input_nc = 18+3
        output_nc = 3
        norm_layer = nn.InstanceNorm2d
        activation = nn.ReLU(True)
        padding_type='reflect'
        
        ###### global generator model #####           
        ngf = 64
        self.model = FgFeature2RGB(ngf=ngf*2)        

        ###### local enhancer layers #####
        ngf_global = ngf
        n_blocks_local = 2
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]   

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
