import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))  
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 5
    opt.niter_decay = 0
    opt.max_dataset_size = 10
    opt.niter_fix_global = 0

### ignore warning
import warnings
warnings.filterwarnings("ignore")

####################### data preparation #######################
def CreateDataset(opt):
    dataset = None
    from data.pre_tex_dataset2 import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

from data.base_data_loader import BaseDataLoader
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        if opt.phase == "train":
            ### split train and validation
            self.ratio = opt.data_ratio
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            # np.random.shuffle(indices)
            split = int(self.ratio * dataset_size)
            train_indices, val_indices = indices[:split], indices[split:]
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

            self.train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=train_sampler,
                num_workers=int(opt.nThreads))
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                sampler=valid_sampler,
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self.train_dataloader, self.valid_dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

data_loader = CreateDataLoader(opt)
train_dataset, val_dataset = data_loader.load_data()
dataset_size = int(len(data_loader)*data_loader.ratio)
val_dataset_size = int(len(data_loader)*(1-data_loader.ratio))
print('#training images = %d' % dataset_size)
print('#validation images = %d' % val_dataset_size)

####################### creat pre-train model #######################
import models.networks as networks
from models.base_model import BaseModel
class TexG(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # TexG_input_nc = opt.input_nc + opt.num_class*3
        TexG_input_nc = opt.input_nc
        self.TexG = networks.define_G(TexG_input_nc, opt.out_tex_nc, opt.ngf_global, opt.TexG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, add_mask=self.opt.use_mask_texture)
        params = list(self.TexG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        if not self.isTrain or opt.continue_train or opt.load_pretrain_TexG:
            pretrained_path = './checkpoint' if not self.isTrain else opt.load_pretrain_TexG
            self.load_network(self.TexG, 'TexG', opt.which_epoch, pretrained_path)

        self.loss_names = ["texture_loss", "mask_loss"] # add bg mask loss
        self.criterion_tex = torch.nn.L1Loss()
        # self.criterion_mask = torch.nn.BCEWithLogitsLoss()
        self.criterion_mask = torch.nn.BCELoss()

    def save(self, which_epoch):
        self.save_network(self.TexG, 'TexG', which_epoch, self.gpu_ids)

    # only for single tensor or list of tensors 
    def encode_input(self, origin):
        if isinstance(origin, list):
            encoded = []
            for item in origin:
                encoded.append(self.encode_input(item))
        else:
            encoded = Variable(origin.data.cuda())
        return encoded

    def forward(self, pose, part_tex_real, mask_real, initial_tex):
        pose, part_tex_real, mask_real = self.encode_input([pose, part_tex_real, mask_real])
        initial_tex = self.encode_input(initial_tex)

        # TexG_input = torch.cat([initial_tex,pose], dim=1)
        TexG_input = pose

        residual, mask =self.TexG(TexG_input)
        # print("initial_tex:", initial_tex.max(), initial_tex.min()) # (0,1)
        # print("residual:", residual.max(), residual.min()) # (-1,1)
        # print("mask:", mask.max(), mask.min()) # (0,1)

        bs, tex_c, tex_h, tex_w = mask.shape
        unsqueezed_mask = mask[:,:,None,...].repeat(1,1,3,1,1).view(bs,-1,tex_h,tex_w)

        # gen_tex = residual + unsqueezed_mask * initial_tex
        gen_tex = (residual + initial_tex) * unsqueezed_mask

        tex_loss = self.criterion_tex(gen_tex, part_tex_real) * opt.lambda_feat * 5
        mask_loss = self.criterion_mask(mask, mask_real) * 5

        return [tex_loss,mask_loss], gen_tex, mask, residual


# for i, data in enumerate(train_dataset):
#     for (k,v) in data.items():
#         if isinstance(v, str):
#             print(k, v)
#         else:
#             print(k, v.shape)
#     input()

####################### training #######################
model = TexG()
model.initialize(opt)
optimizer_G = model.optimizer_G

visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * int(dataset_size*data_loader.ratio) + epoch_iter
valid_step = (start_epoch-1) * int(dataset_size*(1-data_loader.ratio)) + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(train_dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, gen_tex, mask, residual = model(data['pose'], data['part_tex'], data['mask'], data['initial_tex'])

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_G = loss_dict['texture_loss'] + loss_dict['mask_loss']

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()       
        optimizer_G.step()
      

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            visuals = OrderedDict()
            visuals['pose1'] = util.tensor2im(data['pose'][0,:3])
            visuals['pose2'] = util.tensor2im(data['pose'][0,3:6])
            visuals['pose3'] = util.tensor2im(data['pose'][0,6:9])

            visuals['pose_texture'] = util.visualizeTex(data['pose'][0,9:])
            visuals['synthesized_texture'] = util.visualizeTex(gen_tex.data[0])
            visuals['synthesized_tex_mask'] = util.visualizeTex(mask.data[0])
            visuals['synthesized_tex_res'] = util.visualizeTex(residual.data[0])
            visuals['initial_tex'] = util.visualizeTex(data['initial_tex'][0])
            visuals['real_mask'] = util.visualizeTex(data['mask'][0])
            visuals['real_tex'] = util.visualizeTex(data['part_tex'][0])

            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    ############# validation #############
    if not opt.debug:
        model.eval()
        for i, data in enumerate(val_dataset, start=epoch_iter):
            valid_step += opt.batchSize
            with torch.no_grad():
                losses, gen_tex, mask, residual = model(data['pose'], data['part_tex'], data['mask'], data['initial_tex'])
                
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            val_loss_names = ['val'+x for x in model.loss_names]
            loss_dict = dict(zip(val_loss_names, losses))

            if valid_step % val_dataset_size == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                visualizer.plot_current_errors(errors, valid_step)

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')