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
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

### ignore warning
import warnings
warnings.filterwarnings("ignore")

##### create dataset ###
from data.aligned_dataset import *
class DatasetTrans(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.isTrain = opt.isTrain
        ##### load all the json to memory #####
        self.oriH = 1024
        self.oriW = 1024
        poseptsAll = []
        poselen = [54, 69, 75]
        # scale, translate = util.scale_resize((self.oriH,self.oriW),(opt.loadSize, opt.loadSize), mean_height=0.0)
        scale, translate = util.scale_resize((self.oriH,self.oriW), (1024,1024,3), mean_height=0.0)
        self.loadSize = opt.loadSize

        ### input A (pose images)
        self.dir_A = opt.pose_path
        self.A_paths = sorted(make_dataset(self.dir_A))
        # print(self.A_paths)
        if self.isTrain:
            for A_path in self.A_paths:
                ptsList = util.readkeypointsfile_json(A_path)
                if not len(ptsList[0]) in poselen:
                    print("bad json file with %d posepts ..." % len(ptsList))
                    ptsList = poseptsAll[-1]

                # posepts = util.fix_scale_coords(posepts, scale, translate)
                ptsList = [util.fix_scale_coords(xx, scale, translate) for xx in ptsList]
                ptsList = dataArgument(ptsList, xDirection=5, yDirection=20)
                poseptsAll.append(ptsList)
            self.posepts = np.stack(poseptsAll)

        if opt.isTrain:
            self.dir_inst = opt.mask_path
            self.inst_paths = sorted(make_dataset(self.dir_inst))
            inst = Image.open(self.inst_paths[0]).convert('L')
            params = get_params(self.opt, inst.size, inst.mode)
            self.inst_transform = get_transform(self.opt, params, normalize=False) # set to [0,1]

        if opt.isTrain:
            self.dir_C = opt.densepose_path
            self.C_paths = sorted(make_dataset(self.dir_C))

        self.dataset_size = len(self.posepts)
        if opt.isTrain:
            assert(len(self.inst_paths) == self.dataset_size), "mask image is %s while json is %s" % (len(self.inst_paths), self.dataset_size)
            assert(len(self.C_paths) == self.dataset_size), "densepose image is %s while json is %s" % (len(self.C_paths), self.dataset_size)

    def getOpenpose(self, index):
        # posepts = self.posepts[index]
        ptsList = self.posepts[index]
        A = util.renderpose25(ptsList[0], 255 * np.ones((1024,1024,3), dtype='uint8')) # pose
        A = util.renderface_sparse(ptsList[1], A, numkeypoints=8, disp=False)
        A = util.renderhand(ptsList[2], A)
        A = util.renderhand(ptsList[3], A) # [h, w, 3]
        A = cv2.resize(A, (self.loadSize, self.loadSize))
        A_tensor = torch.tensor(A/255).float()*2-1 # [-1,1]
        A_tensor = A_tensor.permute(2,0,1) # [c,h,w]
        return A_tensor

    def getDensepose(self, index, h, w):
        C_path = self.C_paths[index]
        iuv = cv2.imread(C_path).astype(np.float).transpose(2,0,1) # [3, h, w]
        pa, pc = get_parts(iuv)
        pc = pc / 255. * 2 - 1 # set to [-1,1]
        pa = torch.from_numpy(pa).to(torch.float32)
        pc = torch.from_numpy(pc).to(torch.float32)
        # resize
        pa = torch.nn.functional.upsample(pa[None,None,...], size=(h,w), mode='bilinear')
        pc = torch.nn.functional.upsample(pc.unsqueeze(0), size=(h,w), mode='bilinear')
        pa = pa.squeeze()
        pc = pc.squeeze()
        return pa, pc

    def getMask(self, index):
        inst_path = self.inst_paths[index]
        inst = Image.open(inst_path).convert('L')
        inst_tensor = self.inst_transform(inst)
        return inst_tensor

    def __getitem__(self, index):

        index = index % self.dataset_size
        A_tensor = self.getOpenpose(index)
        inst_tensor = self.getMask(index)
        h, w = inst_tensor.shape[-2:]
        pa, pc = self.getDensepose(index, h, w)
        input_dict = {'Pose': A_tensor, 'pa': pa, 'pc': pc, 'mask': inst_tensor}
        return input_dict
    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize
dataset = DatasetTrans()
dataset.initialize(opt)
indices = list(range(len(dataset)))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
train_dataset = torch.utils.data.DataLoader(dataset,
                batch_size=opt.batchSize,
                shuffle=True,
                num_workers=int(opt.nThreads))
                                # sampler=train_sampler,
# data_loader = CreateDataLoader(opt)

# train_dataset, val_dataset = data_loader.load_data()
dataset_size = len(dataset)
print('#training images = %d' % int(dataset_size))
# print('#validation images = %d' % int(dataset_size * (1-data_loader.ratio)))

##### creat pre-train model ###
import models.networks as networks
from models.base_model import BaseModel
class Pix2PixHD_Trans(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        TransG_input_nc = opt.input_nc
        self.TransG = networks.define_G(TransG_input_nc, (opt.num_class*2,opt.num_class+1), opt.ngf_translate, opt.TransG,
                                        opt.n_downsample_translate, opt.n_blocks_translate, opt.n_local_enhancers,
                                        opt.norm, gpu_ids=self.gpu_ids)
        params = list(self.TransG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = './checkpoint' if not self.isTrain else opt.load_pretrain
            self.load_network(self.TransG, 'TransG', opt.which_epoch, pretrained_path)

        self.loss_names = ["UV_loss", "Probs_loss", "mask_loss"] # add bg mask loss
        self.criterion_UV = torch.nn.L1Loss(reduction='none')
        self.criterion_Prob = torch.nn.CrossEntropyLoss()
        # self.criterion_mask = torch.nn.BCEWithLogitsLoss()
        self.criterion_mask = torch.nn.BCELoss()

    def save(self, which_epoch):
        self.save_network(self.TransG, 'TransG', which_epoch, self.gpu_ids)

    # only for single tensor or list of tensors 
    def encode_input(self, origin):
        if isinstance(origin, list):
            encoded = []
            for item in origin:
                encoded.append(self.encode_input(item))
        else:
            encoded = Variable(origin.data.cuda())
        return encoded

    def forward(self, pose, pa_gt, pc_gt, mask):
        pose, pa_gt, pc_gt, mask = self.encode_input([pose, pa_gt, pc_gt, mask])
        pred_UV, pred_Probs =self.TransG(pose)
        # generate UV_mask stack
        UV_mask=[]
        for part_id in range(24):
            UV_mask.append(pa_gt==(part_id+1))
            UV_mask.append(pa_gt==(part_id+1))
        UV_mask_tensor=torch.stack(UV_mask,dim=1)
        UV_mask_tensor=UV_mask_tensor.float()
        UV_loss = self.criterion_UV(pred_UV, pc_gt) * UV_mask_tensor * 500 # mask L1 loss
        Prob_loss = self.criterion_Prob(pred_Probs, pa_gt.long())# mask Prob loss

        norm_Probs = torch.nn.functional.softmax(pred_Probs, dim=1)
        # print(1-norm_Probs[:,0,:,:])
        # print(mask[:,0,:,:])
        # input()
        mask_loss = self.criterion_mask(1-norm_Probs[:,0,:,:], mask[:,0,:,:])

        return [UV_loss,Prob_loss,mask_loss], pred_UV, pred_Probs

##### creat pre-train model END ###

model = Pix2PixHD_Trans()
model.initialize(opt)
optimizer_G = model.optimizer_G

visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * int(dataset_size) + epoch_iter
# valid_step = (start_epoch-1) * int(dataset_size*(1-data_loader.ratio)) + epoch_iter

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
        losses, pred_UV, pred_Probs = model(data['Pose'], data['pa'], data['pc'], data['mask'])

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_G = loss_dict['UV_loss'] + loss_dict['Probs_loss'] + loss_dict['mask_loss']

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
            visuals['pose1'] = util.tensor2im(data['Pose'][0,:3])
            # visuals['pose2'] = util.tensor2im(data['Pose'][0,3:6])
            # visuals['pose3'] = util.tensor2im(data['Pose'][0,6:9])
            im_Probs, im_Probs_GT = util.draw_part_assign(pred_Probs.data[0], data['pa'][0])
            visuals['Probs'] = im_Probs
            visuals['Probs_GT'] = im_Probs_GT
            # print(im_Probs.shape, im_Probs_GT.shape)
            im_U, im_V = util.draw_uv_coordinate(pred_UV.data[0], pred_Probs.data[0])
            visuals['U'] = im_U
            visuals['V'] = im_V
            # print(im_U.shape, im_V.shape)
            im_U_GT, im_V_GT = util.draw_uv_coordinate(data['pc'][0], data['pa'][0])
            visuals['U_GT'] = im_U_GT
            visuals['V_GT'] = im_V_GT
            # print(im_U_GT.shape, im_V_GT.shape)
            # visuals['real_image'] = util.tensor2im(data['real'][0])

            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
    
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

    # ### instead of only training the local enhancer, train the entire network after certain iterations
    # if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
    #     model.update_fixed_params()

    # ### linearly decay learning rate after certain iterations
    # if epoch > opt.niter:
    #     model.update_learning_rate()
