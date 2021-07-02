import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import util.util as util

# from [3,h,w] to pa: [h,w] and pc: [48,h,w]
def get_parts(iuv):
    h, w = iuv.shape[1:]
    pc = np.zeros((48, h, w))
    pa = iuv[0].astype(np.int)
    # assigned to multiple channels
    for i in range(24):
        mask = (pa == (i+1)).astype(np.int)
        pc[i * 2] = mask * iuv[1]
        pc[i * 2 + 1] = mask * iuv[2]
    return pa, pc

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

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def addConfidence(posepts):
    kpsNum, frameNum, _ = posepts.shape
    confidence = np.ones((kpsNum, frameNum, 1))
    posepts = np.concatenate([posepts, confidence], axis=2)
    return posepts

def changeNumpy(posepts):
    return np.array(posepts).reshape(-1,3)

def changeList(posepts):
    return posepts.reshape(-1,).tolist()

def mergePose(posepts, poseptsTgt, facepts, r_handpts, l_handpts):
    posepts = changeNumpy(posepts)
    r_handpts = changeNumpy(r_handpts)
    l_handpts = changeNumpy(l_handpts)
    facepts = changeNumpy(facepts)

    headDiff = posepts[15:19, :2] - posepts[0:1, :2]
    leftFoot = posepts[22:25, :2] - posepts[11:12, :2]
    rightFoot = posepts[19:22, :2] - posepts[14:15, :2]

    # r_handDiff = poseptsTgt[4:5,:2] - posepts[4:5,:2]
    # l_handDiff = poseptsTgt[7:8,:2] - posepts[7:8,:2]
    r_handDiff = poseptsTgt[4:5,:2] - r_handpts[0:1,:2]
    l_handDiff = poseptsTgt[7:8,:2] - l_handpts[0:1,:2]
    faceDiff = poseptsTgt[0:1,:2] - posepts[0:1,:2]

    posepts[:15, :2] = poseptsTgt
    posepts[15:19, :2] = posepts[0:1, :2] + headDiff
    posepts[22:25, :2] = posepts[11:12, :2] + leftFoot
    posepts[19:22, :2] = posepts[14:15, :2] + rightFoot

    r_handpts[:,:2] += r_handDiff
    l_handpts[:,:2] += l_handDiff
    facepts[:,:2] += faceDiff

    return changeList(posepts), changeList(facepts), changeList(r_handpts), changeList(l_handpts)

# def calDiff(pts, base):

def aveConfidence(ptsList):
    assert len(ptsList) % 3 == 0
    Num = len(ptsList)//3
    conf = 0
    for idx in range(Num):
        conf += ptsList[idx*3+2]
    return conf/Num

def getConsequentList(indexList):
    # [1,2,6,7,8,14,15] -> [[1,2],[6,7,8],[14,15]], [[0,1],[2,3,4],[5,6]]
    if len(indexList) == 0:
        return indexList, []
    if len(indexList) == 1:
        return [indexList], [[0]]
    ret, retID = [], []
    idx = 0
    while idx < len(indexList)-1:
        curRet, curRetID = [], []
        while idx < len(indexList)-1 and indexList[idx+1]-indexList[idx] == 1:
            curRet.append(indexList[idx])
            curRetID.append(idx)
            idx += 1
        curRet.append(indexList[idx])
        curRetID.append(idx)
        idx += 1
        ret.append(curRet)
        retID.append(curRetID)
    return ret, retID
        

def interpolatePose(kps, threshold=0.05):
    # [NumFrame, NumKp], np.array
    NumFrame = len(kps)
    # import ipdb; ipdb.set_trace()
    for kpID in range(kps.shape[1]):
        frameIDs = np.where(kps[:,kpID] <= 0)[0]
        consecFrames, consecIndexs = getConsequentList(frameIDs)
        if len(consecFrames) == 0:
            continue
        for consecFrame, consecIndex in zip(consecFrames,consecIndexs):
            Num = len(consecFrame)
            startFrameID = consecFrame[0]-1
            endFrameID = consecFrame[-1]+1
            if startFrameID < 0:
                start = kps[endFrameID][kpID]
                end = kps[endFrameID][kpID]
            elif endFrameID >= NumFrame:
                start = kps[startFrameID][kpID]
                end = kps[startFrameID][kpID]
            else:
                start = kps[startFrameID][kpID]
                end = kps[endFrameID][kpID]

            for idx in range(startFrameID+1, endFrameID):
                ratio = (idx-startFrameID)/Num
                kps[idx][kpID] = ratio * start + (1-ratio) * end
    return kps

def interpolatePose2(kps, threshold=0.05):
    # [NumFrame, NumKp, 3], np.array
    NumFrame = len(kps)
    for kpID in range(kps.shape[1]):
        frameIDs = np.where(kps[:,kpID,2] <= threshold)[0]
        consecFrames, consecIndexs = getConsequentList(frameIDs)
        if len(consecFrames) == 0:
            continue
        for consecFrame, consecIndex in zip(consecFrames,consecIndexs):
            Num = len(consecFrame)
            startFrameID = consecFrame[0]-1
            endFrameID = consecFrame[-1]+1
            if startFrameID < 0:
                start = kps[endFrameID][kpID]
                end = kps[endFrameID][kpID]
            elif endFrameID >= NumFrame:
                start = kps[startFrameID][kpID]
                end = kps[startFrameID][kpID]
            else:
                start = kps[startFrameID][kpID]
                end = kps[endFrameID][kpID]

            for idx in range(startFrameID+1, endFrameID):
                ratio = (idx-startFrameID)/Num
                kps[idx][kpID] = (1-ratio) * start + ratio * end
    return kps

from scipy.ndimage import gaussian_filter1d
def smoothPoseList(poseList):
    # [[posekps, facekps, rhandkps, lhandkps]] * Num
    Num = len(poseList)
    posekps, facekps, rhandkps, lhandkps = [], [], [], []
    for idx in range(Num):
        posekps.append(changeNumpy(poseList[idx][0]))
        facekps.append(changeNumpy(poseList[idx][1]))
        rhandkps.append(changeNumpy(poseList[idx][2]))
        lhandkps.append(changeNumpy(poseList[idx][3]))

    posekps = np.stack(posekps) # [Num, 25, 3]
    facekps = np.stack(facekps)
    rhandkps = np.stack(rhandkps)
    lhandkps = np.stack(lhandkps)

    posekps = interpolatePose2(posekps) # process three axis the same time

    posekps[:,:,:2] = gaussian_filter1d(posekps[:,:,:2], sigma=2, axis=0)
    facekps[:,:,:2] = gaussian_filter1d(facekps[:,:,:2], sigma=2, axis=0)
    rhandkps[:,:,:2] = gaussian_filter1d(rhandkps[:,:,:2], sigma=2, axis=0)
    lhandkps[:,:,:2] = gaussian_filter1d(lhandkps[:,:,:2], sigma=2, axis=0)

    for idx in range(Num):
        poseList[idx][0] = changeList(posekps[idx])
        poseList[idx][1] = changeList(facekps[idx])
        poseList[idx][2] = changeList(rhandkps[idx])
        poseList[idx][3] = changeList(lhandkps[idx])

    return poseList

def calDiffwithBase(pts, base):
    ret = pts.copy()
    ret[:,:,:2] = pts[:,:,:2] - base[:,:,:2]
    return ret

def addDifftoBase(base, diff):
    diff[:,:,:2] += base[:,:,:2]
    return diff

def interpolateHand(allRhandpts, allbase):
    # [NumFrame,21,3], [NumFrame,1,3]
    # process all the kpID at the same time
    import ipdb; ipdb.set_trace()
    pass

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:,np.newaxis]

def degree2rad(degree):
    return degree/180*np.pi
# 
def genHandkps(limb, unit=25, isRight=True):
    # [N,2,3]: number 3 and limb, unit310)4
        # [N,2,3]: number 3 and limb, unit310)4 of openpose skeleton of openpose skeleton
    N = limb.shape[0]
    handpts = np.zeros((N,21,2))
    handpts[:,0] = limb[:,1,:2]
    direction = normalize(limb[:,1,:2] - limb[:,0,:2]) # [N,2]
    isLeft = not isRight
    # third
    handpts[:,9] = handpts[:,0] + direction * unit*0.8
    handpts[:,10] = handpts[:,9] + direction * unit*0.8
    handpts[:,11] = handpts[:,10] + direction * unit*0.4
    handpts[:,12] = handpts[:,11] + direction * unit*0.2
    # first
    rad = degree2rad(40)
    if isLeft: 
        rad *= -1
    rotateMat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    curDirection = np.array([rotateMat@direction[i] for i in range(len(direction))])
    handpts[:,1] = handpts[:,0] + curDirection * unit*0.5
    handpts[:,2] = handpts[:,1] + curDirection * unit*0.5
    rad = degree2rad(-20)
    if isLeft: 
        rad *= -1
    rotateMat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    curDirection = np.array([rotateMat@curDirection[i] for i in range(len(curDirection))])
    handpts[:,3] = handpts[:,2] + curDirection * unit*0.3
    handpts[:,4] = handpts[:,3] + curDirection * unit*0.1
    # second
    rad = degree2rad(15)
    if isLeft: 
        rad *= -1
    rotateMat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    curDirection = np.array([rotateMat@direction[i] for i in range(len(direction))])
    handpts[:,5] = handpts[:,0] + curDirection * unit*0.8
    handpts[:,6] = handpts[:,5] + curDirection * unit*0.6
    handpts[:,7] = handpts[:,6] + direction * unit*0.4
    handpts[:,8] = handpts[:,7] + direction * unit*0.2
    # fourth
    rad = degree2rad(-15)
    if isLeft: 
        rad *= -1
    rotateMat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    curDirection = np.array([rotateMat@direction[i] for i in range(len(direction))])
    handpts[:,13] = handpts[:,0] + curDirection * unit*0.8
    handpts[:,14] = handpts[:,13] + curDirection * unit*0.6
    handpts[:,15] = handpts[:,14] + direction * unit*0.4
    handpts[:,16] = handpts[:,15] + direction * unit*0.2
    # fifth
    rad = degree2rad(-35)
    if isLeft: 
        rad *= -1
    rotateMat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    curDirection = np.array([rotateMat@direction[i] for i in range(len(direction))])
    handpts[:,17] = handpts[:,0] + curDirection * unit*0.6
    handpts[:,18] = handpts[:,17] + curDirection * unit*0.5
    rad = degree2rad(20)
    if isLeft: 
        rad *= -1
    rotateMat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    curDirection = np.array([rotateMat@curDirection[i] for i in range(len(curDirection))])
    handpts[:,19] = handpts[:,18] + curDirection * unit*0.4
    handpts[:,20] = handpts[:,19] + curDirection * unit*0.3

    handConf = np.ones((N,21,1))
    handpts = np.concatenate([handpts,handConf], axis=2)

    handpts[np.isnan(handpts)] = 0

    return handpts

def processFoot(posekpt):
    # posekpt: list of 25x3
    posekpt = changeNumpy(posekpt)[np.newaxis,...] # [1,25,3]
    leftFoot = calDiffwithBase(posekpt[:, 22:25], posekpt[:, 11:12])
    rightFoot = calDiffwithBase(posekpt[:, 19:22], posekpt[:, 14:15])
    footLen = 50
    leftFoot[:,0,:2] = footLen * normalize(leftFoot[:,0,:2])
    rightFoot[:,0,:2] = footLen * normalize(rightFoot[:,0,:2])
    posekpt[:, 22:25] = addDifftoBase(posekpt[:, 11:12], leftFoot)
    posekpt[:, 19:22] = addDifftoBase(posekpt[:, 14:15], rightFoot)
    return changeList(posekpt)


def mergeBatchPose(allPosepts, allPoseptsTgt, allFacepts, allRhandpts, allLhandpts):

    headDiff = calDiffwithBase(allPosepts[:, 15:19], allPosepts[:, 0:1])
    leftFoot = calDiffwithBase(allPosepts[:, 22:25], allPosepts[:, 11:12])
    rightFoot = calDiffwithBase(allPosepts[:, 19:22], allPosepts[:, 14:15])

    ### interpolate ###
    headDiff, leftFoot, rightFoot = tuple(map(interpolatePose2, [headDiff, leftFoot, rightFoot]))

    allRhandpts = genHandkps(allPosepts[:, 3:5])
    allLhandpts = genHandkps(allPosepts[:, 6:8], isRight=False)

    RhandDiff = calDiffwithBase(allRhandpts, allPosepts[:, 4:5])
    LhandDiff = calDiffwithBase(allLhandpts, allPosepts[:, 7:8])
    faceDiff = calDiffwithBase(allFacepts, allPosepts[:, 0:1])


    allPosepts[:, :15, :2] = allPoseptsTgt

    allPosepts[:, 15:19] = addDifftoBase(allPosepts[:, 0:1], headDiff)
    footLen = 50

    allPosepts[:, 22:25] = addDifftoBase(allPosepts[:, 11:12], leftFoot)
    allPosepts[:, 19:22] = addDifftoBase(allPosepts[:, 14:15], rightFoot)

    allPosepts[np.isnan(allPosepts)] = 0

    allRhandpts = addDifftoBase(allPosepts[:, 4:5], RhandDiff)
    allLhandpts = addDifftoBase(allPosepts[:, 7:8], LhandDiff)
    allFacepts = addDifftoBase(allPosepts[:, 0:1], faceDiff)

    return allPosepts, allFacepts, allLhandpts, allRhandpts

def dataArgument(ptsList, xDirection, yDirection):

    permute = np.zeros([1,2])
    if np.random.rand() > 0.5:
        permute[0,0] = xDirection * np.random.randn()
        permute[0,1] = yDirection * np.random.randn()
    for idx, x in enumerate(ptsList):
        x = changeNumpy(x)
        x[:,:2] = x[:,:2] + permute
        ptsList[idx] = changeList(x)
    return ptsList

class AlignedDataset(BaseDataset):
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
        USETEST = False
        USETEST2 = True # PoseNorm

        if self.isTrain:
            for A_path in self.A_paths:
                ptsList = util.readkeypointsfile_json(A_path)
                if not len(ptsList[0]) in poselen:
                    print("bad json file with %d posepts ..." % len(ptsList[0]))
                    ptsList = poseptsAll[-1]
                # ptsList[0] = processFoot(ptsList[0])
                # posepts = util.fix_scale_coords(posepts, scale, translate)
                ptsList = [util.fix_scale_coords(xx, scale, translate) for xx in ptsList]
                # ptsList = dataArgument(ptsList, xDirection=5, yDirection=20)
                poseptsAll.append(ptsList)
            self.posepts = np.stack(poseptsAll)
        elif USETEST:
            ################ transfer skeleton with this approach: https://github.com/ChrisWu1997/2D-Motion-Retargeting
            posepts = np.load(opt.npz_path)['out12'].transpose(2,0,1) # (15, 2, Num) -> (Num, 15, 2)
            # posepts = addConfidence(posepts)
            allPosepts, allFacepts, allRhandpts, allLhandpts = [], [], [], []
            allPoseptsTgt = posepts
            for idx, x in enumerate(posepts):
                srcposepts, facepts, r_handpts, l_handpts = util.readkeypointsfile_json(self.A_paths[idx])
                allPosepts.append(changeNumpy(srcposepts))
                allFacepts.append(changeNumpy(facepts))
                allRhandpts.append(changeNumpy(r_handpts))
                allLhandpts.append(changeNumpy(l_handpts))
            
            allPosepts, allFacepts, allLhandpts, allRhandpts = \
                mergeBatchPose(np.stack(allPosepts), posepts, np.stack(allFacepts), np.stack(allRhandpts), np.stack(allLhandpts))

            for posepts, facepts, lhandpts, rhandpts in zip(allPosepts, allFacepts, allLhandpts, allRhandpts):
                poseptsAll.append([changeList(posepts), changeList(facepts), changeList(lhandpts), changeList(rhandpts)])

            self.posepts = smoothPoseList(poseptsAll)
        elif USETEST2:
            ################ transfer skeleton by alignling pose manually
            from data.data_prep.graph_posenorm import PoseNorm
            dance15Tuple = (800,1000)
            dance15Tuple = (1000,1500)
            allPosepts, allFacepts, allRhandpts, allLhandpts = PoseNorm(opt.pose_path, opt.pose_tgt_path, (0,50), dance15Tuple, (1024,1024,3), (1024,1024,3))
            ### modfied ###
            Num = allPosepts.shape[0]
            allPosepts = allPosepts.reshape(Num,25,3)
            allPosepts[:,:,1] = allPosepts[:,:,1] - 0 # move person skeleton up/down manually
            allPosepts = allPosepts.reshape(Num,-1)
            # import ipdb; ipdb.set_trace()
            allRhandpts = genHandkps(allPosepts.reshape(Num,25,3)[:, 3:5]).reshape(Num,-1)
            allLhandpts = genHandkps(allPosepts.reshape(Num,25,3)[:, 6:8], isRight=False).reshape(Num,-1)
            for posepts, facepts, lhandpts, rhandpts in zip(allPosepts, allFacepts, allLhandpts, allRhandpts):
                poseptsAll.append([posepts, facepts, lhandpts, rhandpts])
            # self.posepts = np.stack(poseptsAll)
            self.posepts = smoothPoseList(poseptsAll)
        else:
            for A_path in self.A_paths:
                ptsList = util.readkeypointsfile_json(A_path)
                if not len(ptsList[0]) in poselen:
                    print("bad json file with %d posepts ..." % len(ptsList[0]))
                    ptsList = poseptsAll[-1]
                # ptsList[0] = processFoot(ptsList[0])
                # posepts = util.fix_scale_coords(posepts, scale, translate)
                ptsList = [util.fix_scale_coords(xx, scale, translate) for xx in ptsList]
                poseptsAll.append(ptsList)

            self.posepts = np.stack(poseptsAll)[-200:]

        ### input A (pose texture images)
        self.dir_A_tex = opt.pose_texture_path
        if os.path.exists(self.dir_A_tex):
            self.A_tex_paths = sorted(make_dataset(self.dir_A_tex))
        else:
            print("No pose texture!")
            self.A_tex_paths = None

        ### input B (real images)
        if opt.isTrain:
            self.dir_B = opt.img_path
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### input C (densepose images)
        if opt.isTrain:
            self.dir_C = opt.densepose_path
            self.C_paths = sorted(make_dataset(self.dir_C))

        ### densepose mask
        if opt.isTrain:
            self.dir_inst = opt.mask_path
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### set transform
        if opt.isTrain:
            B = Image.open(self.B_paths[0]).convert('RGB')
            params = get_params(self.opt, B.size, B.mode)
            self.transform = get_transform(self.opt, params)

        if opt.isTrain:
            inst = Image.open(self.inst_paths[0]).convert('L')
            params = get_params(self.opt, inst.size, inst.mode)
            self.inst_transform = get_transform(self.opt, params, normalize=False) # set to [0,1]

        ### texture image and transform
        tex_size = self.tex_size = opt.tex_size
        self.texture = 0
        if opt.isTrain:
            texture = Image.open(opt.texture_path).convert('RGB')
            if texture.width // 4 != self.tex_size:
                texture = texture.resize((self.tex_size*4, self.tex_size*6))
            tex_transform = [transforms.ToTensor()] # set to [0,1]
            # , transforms.Normalize((0.5,)*3, (0.5,)*3) # set to [-1,1]
            self.tex_transform = transforms.Compose(tex_transform)
            texture = self.tex_transform(texture) # [3, h, w]

            self.texture = tex_im2tensor(texture, self.tex_size)

        ### background and transform
        self.bg = 0
        if opt.isTrain:
            bg = Image.open(opt.bg_path).convert('RGB')
            bg_transform = [transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)] # set to [-1,1]
            bg_transform = transforms.Compose(bg_transform)
            bg = bg_transform(bg) # [3, bg_h, bg_w]
            self.bg = torch.nn.functional.upsample(bg.unsqueeze(0), size=opt.loadSize, mode='bilinear')[0] # [3, h, w]

        self.dataset_size = len(self.posepts)
        if opt.isTrain:
            assert(len(self.inst_paths) == self.dataset_size)
            assert(len(self.B_paths) == self.dataset_size)

        ### optical flow
        if opt.isTrain:
            self.dir_flow = opt.flow_path
            self.flow_paths = sorted(make_dataset(self.dir_flow))

            self.dir_flow_inv = opt.flow_inv_path
            self.flow_inv_paths = sorted(make_dataset(self.dir_flow_inv))

    def getOpenpose(self, index):
        # posepts = self.posepts[index]
        ptsList = self.posepts[index]
        A = util.renderpose25(ptsList[0], 255 * np.ones((1024,1024,3), dtype='uint8')) # pose
        # A = util.renderface_sparse(ptsList[1], A, numkeypoints=8, disp=False)
        A = util.renderhand(ptsList[2], A, threshold = 0.05)
        A = util.renderhand(ptsList[3], A, threshold = 0.05) # [h, w, 3]
        A = cv2.resize(A, (self.loadSize, self.loadSize))
        A_tensor = torch.tensor(A/255).float()*2-1 # [-1,1]
        A_tensor = A_tensor.permute(2,0,1) # [c,h,w]
        return A_tensor

    def getImage(self, index):
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        B_tensor = self.transform(B)
        return B_tensor

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

    def getFlow(self, index, inv=False):
        if inv:
            flow_path = self.flow_inv_paths[index-1] # the flow between current and before
        else:
            flow_path = self.flow_paths[index-1] # the flow between current and before
        flow = readFlow(flow_path).transpose(2,0,1) # change to (c,h,w)
        flow = torch.tensor(flow) # (2, original_h, original_w)
        if flow.shape[-2] != self.loadSize or flow.shape[-1] != self.loadSize:
            print("the pre-computed flow has different size ... ")
            flow = torch.nn.functional.upsample(flow.unsqueeze(0), size=self.loadSize, mode='bilinear')
            flow = flow[0]
        return flow


    def __getitem__(self, index):
        if self.opt.isTrain and index == 0:
            index += 1 # process the start frame

        index = index % self.dataset_size
        A_tensor = self.getOpenpose(index)
        if self.isTrain:
            A_tensor_before = self.getOpenpose(index-1)

        if self.isTrain:
            A_path = self.A_paths[index]
            A_path_before = self.A_paths[index-1]


        B_tensor = pa = pc = 0
        ### input B (real images)
        if self.opt.isTrain:
            B_tensor = self.getImage(index)
            if self.isTrain:
                B_tensor_before = self.getImage(index-1)

            h, w = B_tensor.shape[-2:]

            pa, pc = self.getDensepose(index, h, w)
            if self.isTrain:
                pa_before, pc_before = self.getDensepose(index-1, h, w)

        ### densepose mask
        inst_tensor = 0
        if self.opt.isTrain:
            inst_tensor = self.getMask(index)
            inst_tensor_before = self.getMask(index-1)

        # optical flow
        flow = flow_inv = 0
        if self.opt.isTrain:
            flow = self.getFlow(index)
            flow_inv = self.getFlow(index, inv=True)

        if self.isTrain:
            input_dict = {'texture': self.texture, 'Pose': A_tensor, 'mask': inst_tensor,
                        'real': B_tensor, 'path': A_path, 'pa': pa, 'pc': pc, 'bg': self.bg,
                        'Pose_before': A_tensor_before, 'mask_before': inst_tensor_before,
                        'real_before': B_tensor_before, 'path_before': A_path_before, 'pa_before': pa_before, 'pc_before': pc_before,
                        'flow': flow, 'flow_inv': flow_inv}
        else:
            input_dict = {'texture': self.texture, 'Pose': A_tensor, 
                        'real': B_tensor, 'pa': pa, 'pc': pc, 'bg': self.bg}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

if __name__ == "__main__":
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    opt.pose_path = "/home/sunyangtian/data/everybody_data/train_label"
    opt.mask_path = "/home/sunyangtian/data/mask"
    opt.img_path = "/home/sunyangtian/data/dance18"
    opt.densepose_path = "/home/sunyangtian/data/densepose"
    opt.texture_path = "./datasets/texture.jpg"
    opt.bg_path = "./datasets/bg.jpg"
    opt.loadSize = 1080
    dataset = AlignedDataset()
    dataset.initialize(opt)

    data = dataset[0]
    for (k,v) in data.items():
        if isinstance(v, str):
            print(k, v)
        else:
            print(k, v.shape)
    input()
    import util.util as util
    import cv2

    ### visulize texture
    texture = util.visualizeTex(data['texture'])
    cv2.imwrite('./text0624_texture.jpg', texture) # just save RGB, no BGR

    # IUV = cv2.imread('/home/sunyangtian/data/densepose/dance18_00000001_IUV.png').astype(np.float)
    IUV = cv2.imread('/home/sunyangtian/data/densepose/dance18_00000001_IUV.png').astype(np.float)
    IUV[:,:,1:] = IUV[:,:,1:]/255.

    ######### from densepose to UVs and Probs #########
    I = IUV[:,:,0]
    UV = IUV[:,:,1:]*2-1
    UVs = []
    Probs = [np.zeros_like(I)[...,np.newaxis]]
    for idx in range(1,25):
        uv = np.zeros_like(UV)
        uv[I==idx] = UV[I==idx]
        UVs.append(uv)
        prob = np.zeros_like(I)[...,np.newaxis]
        prob[I==idx] = 1.
        Probs.append(prob)
    UVs = np.concatenate(UVs, axis=2).transpose(2,0,1) # [48,h,w]
    Probs = np.concatenate(Probs, axis=2).transpose(2,0,1) # [25,h,w]
    UVs = torch.from_numpy(UVs[np.newaxis,...]).to(torch.float32)
    Probs = torch.from_numpy(Probs[np.newaxis,...]).to(torch.float32)
    ######### from densepose to UVs and Probs #########

    ### texture2image_IUV
    # IUV = torch.from_numpy(IUV).to(torch.float32)
    # image = util.texture2image_IUV(data['texture'].view(24,3,200,200), IUV)

    ### texture2image
    # print(data['pc'].max(), data['pc'].min())
    image = util.texture2image(data['texture'].unsqueeze(0), data['pc'].unsqueeze(0), Probs)
    im_u, im_v = util.draw_uv_coordinate(data['pc'], data['pa'])
    cv2.imwrite('./test0623_u.jpg', im_u)
    cv2.imwrite('./test0623_v.jpg', im_v)
    print("data['texture']:", data['texture'].max(), data['texture'].min())
    print("warp image:", image.max(), image.min())
    cv2.imwrite('./test0618.jpg', (image[0].detach().numpy().transpose(1,2,0)[:,:,::-1])*255)
    print(image.shape)
