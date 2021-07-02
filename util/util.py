from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import numpy as np
import os
import cv2

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# convert [25, h, w] and [h, w] to [h, w]
def draw_part_assign(prediction, GT):
    img1 = torch.argmax(prediction, dim=0) #[h, w]
    img1 = img1.cpu().numpy()
    img1 = img1 / 24. * 255
    img2 = GT.cpu().numpy()
    img2 = img2 / 24. * 255
    return img1.astype(np.uint8), img2.astype(np.uint8)

def draw_part_assign_pred(prediction):
    img1 = torch.argmax(prediction, dim=0) #[h, w]
    img1 = img1.cpu().numpy()
    img1 = img1 / 24. * 255
    return img1.astype(np.uint8)

# convert [48, h, w] and [25,h,w] / [h,w] to [h, w]
def draw_uv_coordinate(uv,probs):
    if probs.shape[0] == 25:
        imID = torch.argmax(probs, dim=0).cpu().numpy()
    else:
        imID = probs.cpu().numpy()
    uv = (uv + 1) / 2 # set to [0,1]
    uv = uv.clamp(0, 1).cpu().numpy()
    h, w = uv.shape[1:]
    # u = v = np.zeros((h, w))
    u = np.zeros((h, w))
    v = np.zeros((h, w))
    for partID in range(24):
        u[imID==(partID+1)] = uv[partID*2][imID==(partID+1)]
        v[imID==(partID+1)] = uv[partID*2+1][imID==(partID+1)]
    return (u*255.).astype(np.uint8), (v*255.).astype(np.uint8)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

############
# convert texture (tensor [24,3,tex_size,tex_size]) and IUV (tensor [h,w,3]) to image
def texture2image_IUV(textureIm, IUV):
    IUV = IUV[None,:,:,:] # [1,h,w,3]
    I = IUV[:,:,:,0]
    UV = IUV[:,:,:,1:]
    genereted_img = torch.zeros_like(IUV).permute(0,3,1,2) # [1,3,h,w]
    partNUM = len(textureIm)
    for partID in range(1, partNUM+1):
        uv_coord = torch.zeros_like(UV)
        uv_coord[I==partID] = UV[I==partID] # [1,h,w,2]
        uv_coord = uv_coord * 2 - 1 # set from [0,1] to [-1,1]
        texture = textureIm[partID-1][None,:,:,:] # [1,3,tex_size,tex_size]
        genereted_img += torch.nn.functional.grid_sample(texture, uv_coord)
    return genereted_img

# convert texture (tensor [bs,24x3,tex_size,tex_size]) and UV [bs,48,h,w] && Prob [bs,25,h,w] to image  
def texture2image(textureIm, UVs, Probs, selNUM=None):
    h, w = UVs.shape[-2:]
    bs = UVs.shape[0]
    if (textureIm.shape[0] != bs):
        textureIm = textureIm.expand(bs,-1,-1,-1)
    device = textureIm.device
    # partNUM = textureIm.shape[1] // 3
    partNUM = 24
    chanNUM = textureIm.shape[1] // partNUM
    if selNUM is None:
        selNUM = chanNUM
    generated_img = torch.zeros([bs, selNUM, h, w]).to(device) # [bs,3,h,w]

    Probs = torch.nn.functional.softmax(Probs, dim=1)

    for partID in range(1, partNUM+1):
        texture = textureIm[:,(partID-1)*chanNUM:(partID-1)*chanNUM+selNUM,:,:] # [bs,3,tex_size,tex_size]
        uv = UVs[:,(partID-1)*2:partID*2,:,:].permute(0,2,3,1) # [bs,h,w,2]
        img = torch.nn.functional.grid_sample(texture, uv) # [bs,3,h,w]
        prob = Probs[:,partID,:,:].unsqueeze(1) # [bs,1,h,w]
        generated_img += img * prob # [bs,3,h,w]
        # generated_img += img # [bs,3,h,w]
    return generated_img

def pickPart(textureIm, partID, selNUM=3):
    # [bs, 24xchanNum, texSize, texSize], ID from 1 to 25
    chanNUM = textureIm.shape[1] // 24
    return textureIm[:,(partID-1)*chanNUM:(partID-1)*chanNUM+selNUM,:,:]

def catTextureTensor(textureIm, addTensor):
    # [bs, 24x3, texSize, texSize] & [bs, 24x15, texSize, texSize]
    chanNUM = addTensor.shape[1] // 24
    partNUM = 24
    newTextureIm = []
    for partID in range(1, partNUM+1):
        newTextureIm.append(torch.cat([pickPart(textureIm, partID), pickPart(addTensor, partID, selNUM=chanNUM)], dim=1))
    return torch.cat(newTextureIm, dim=1)

def pickbaseTexture(textureIm):
    # # [bs, 24xchanNum, texSize, texSize]
    baseTex = []
    partNUM = 24
    for partID in range(1, partNUM+1):
        baseTex.append(pickPart(textureIm, partID))
    return torch.cat(baseTex, dim=1)

def warp(gen_texture, warp_grid):
    '''
        gen_texture: [bs,24x3,tex_size,tex_size]
        warp_grid: [bs,48,tex_size,tex_size]
    '''
    H, W = gen_texture.shape[-2:]
    ygrid, xgrid = np.meshgrid(
            np.linspace(-1.0, 1.0, H),
            np.linspace(-1.0, 1.0, W), indexing='ij')
    grid = torch.tensor(np.stack((xgrid, ygrid), axis=0)[None].astype(np.float32), device=gen_texture.device) # [1, 2, 200, 200]
    grid = grid.permute([0,2,3,1]).repeat(warp_grid.shape[0],1,1,1) # copy to the same batch size
    gen_texture = gen_texture.repeat(warp_grid.shape[0],1,1,1) # copy to the same batch size

    warp_grid = warp_grid.permute(0,2,3,1)
    partNUM = gen_texture.shape[1] // 3
    warped_textures = []
    for partID in range(1, partNUM+1):
        texture = gen_texture[:,(partID-1)*3:partID*3,:,:] # [bs,3,tex_size,tex_size]
        part_grid = grid.clone() + warp_grid[:,:,:,(partID-1)*2:partID*2]
        warped_tex = torch.nn.functional.grid_sample(texture, part_grid)
        warped_textures.append(warped_tex)
    return torch.cat(warped_textures, dim=1)

def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=2.0, skip_amount=50):
    # Don't affect original image
    image = image.copy()
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)
    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])), 2)
    flow_end = (optical_flow_image[flow_start[:,:,1],flow_start[:,:,0],:2] + flow_start).astype(np.int32)
    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt2=tuple(flow_start[y,x]), 
                        pt1=tuple(flow_end[y,x]),
                        color=(0, 255, 0), 
                        thickness=1, 
                        tipLength=.2)
    return image

def vis_tex_flow(warp_grid):
    '''
        warp_grid: [48, 200, 200]
    '''
    H, W = warp_grid.shape[-2:]
    partNUM = warp_grid.shape[0] // 2
    image = np.ones((H,W,3)).astype(np.int32) * 255

    warp_grid = warp_grid.permute(1,2,0)

    part_imgs = []
    visTexture = np.zeros((H*6, W*4, 3), dtype=np.uint8) # (1200, 800)
    for partID in range(1, partNUM+1):
        flow_image = warp_grid[:, :, (partID-1)*2:partID*2].cpu().numpy() # the first of each batch
        flow_image[:,:,0] *= (flow_image.shape[1]/2) # multiply w
        flow_image[:,:,1] *= (flow_image.shape[0]/2) # multiply h
        part_img = put_optical_flow_arrows_on_image(image, flow_image)
        i, j = (partID-1) // 6, (partID-1) % 6
        visTexture[(H*j):(H*j+H), (W*i):(W*i+W), :] = part_img
    return visTexture


def morph(src_bg_mask, ks, mode='erode', kernel=None):
    n_ks = ks ** 2
    pad_s = ks // 2

    if kernel is None:
        kernel = torch.ones(1, 1, ks, ks, dtype=torch.float32, device=src_bg_mask.device)

    if mode == 'erode':
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=1.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out == n_ks).float()
    else:
        src_bg_mask_pad = F.pad(src_bg_mask, [pad_s, pad_s, pad_s, pad_s], value=0.0)
        out = F.conv2d(src_bg_mask_pad, kernel)
        out = (out >= 1).float()

    return out

def visualizeTex(TextureIm):
    c, h, w= TextureIm.shape[-3:]
    c = c // 24 # channal of image
    if c == 1:
        visTexture = np.zeros((h*6, w*4), dtype=np.uint8) # (1200, 800)
        for i in range(4): # x coordinate
            for j in range(6): # y coordinate
                visTexture[(h*j):(h*j+h), (w*i):(w*i+w)] = tensor2im(TextureIm[(6*i+j)*c:(6*i+j+1)*c,torch.arange(h-1,-1,-1),:], normalize=False)
    else:
        visTexture = np.zeros((h*6, w*4, c), dtype=np.uint8) # (1200, 800, 3)
        for i in range(4): # x coordinate
            for j in range(6): # y coordinate
                visTexture[(h*j):(h*j+h), (w*i):(w*i+w), :] = tensor2im(TextureIm[(6*i+j)*c:(6*i+j+1)*c,torch.arange(h-1,-1,-1),:], normalize=False)
    return visTexture # RGB, not BGR


def All_TVloss(UVs, Probs):
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    # x with the shape [bs, c, h, w]
    TVLoss_weight = 3e6

    x = UVs[:,:2,:,:]
    batch_size = x.size()[0]
    c_x = x.size()[1]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])

    PART_NUM = Probs.shape[1]
    Probs = torch.nn.functional.softmax(Probs, dim=1)
    # Probs = torch.nn.functional.softmax(Probs.view(batch_size, PART_NUM, -1), dim=-1).view(batch_size, PART_NUM, h_x, w_x)

    h_tvs, w_tvs = 0, 0
    for idx in range(1,25):
        x = UVs[:,(idx-1)*2:idx*2,:,:] # [bs,h,w,2]


        prob = Probs[:,idx,:,:].unsqueeze(1) # [bs,1,h,w]
        prob = torch.nn.functional.softmax(prob.view(batch_size, 1, -1), dim=-1).view(batch_size, 1, h_x, w_x)

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)
        h_tv = (h_tv * prob[:,:,:-1,:]).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)
        w_tv = (w_tv * prob[:,:,:,:-1]).sum()

        h_tvs += h_tv
        w_tvs += w_tv

    return TVLoss_weight*2*(h_tvs/count_h+w_tvs/count_w)/batch_size


import cv2
import math
def renderpose25(posepts, canvas):
	""" FILL THIS IN """
	colors = [[255,     0,    85], \
		[255,     0,     0], \
		[255,    85,     0], \
		[255,   170,     0], \
		[255,   255,     0], \
		[170,   255,     0], \
		[85,   255,     0], \
		[0,   255,     0], \
		[255,     0,     0], \
		[0,   255,    85], \
		[0,   255,   170], \
		[0,   255,   255], \
		[0,   170,   255], \
		[0,    85,   255], \
		[0,     0,   255], \
		[255,     0,   170], \
		[170,     0,   255], \
		[255,     0,   255], \
		[85,     0,   255], \
		[0,     0,   255], \
		[0,     0,   255], \
		[0,     0,   255], \
		[0,   255,   255], \
		[0,   255,   255], \
		[0,   255,   255]]

	i = 0
	while i < 25*3:
		if i == 23*3 or i == 24*3 or i == 20*3 or i == 21*3:
			i += 3
			continue
		confidence = posepts[i+2]
		if confidence > 0 :
			cv2.circle(canvas, (int(posepts[i]), int(posepts[i+1])), 8, tuple(colors[i // 3]), thickness=-1)
		i += 3

	# limbSeq = [[0,1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], \
	# 		[9, 10], [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 19], [14, 21], [15, 17], [16, 18], \
	# 		[19, 20], [22, 23]]
	limbSeq = [[0,1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], \
			[9, 10], [10, 11], [11, 22], [12, 13], [13, 14], [14, 19], [15, 17], [16, 18]]
	# limbSeq = [[0,1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], \
	# 		[9, 10], [10, 11], [12, 13], [13, 14]]
	# limbSeq = [[0,1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], \
	# 		[9, 10], [10, 11], [11, 22], [11, 24], [12, 13], [13, 14], [14, 21], [15, 17]]

	stickwidth = 4

	for k in range(len(limbSeq)):
		firstlimb_ind = limbSeq[k][0]
		secondlimb_ind = limbSeq[k][1]

		if (posepts[3*firstlimb_ind + 2] > 0) and (posepts[3*secondlimb_ind + 2] > 0):
			cur_canvas = canvas.copy()
			Y = [posepts[3*firstlimb_ind], posepts[3*secondlimb_ind]]
			X = [posepts[3*firstlimb_ind + 1], posepts[3*secondlimb_ind + 1]]
			mX = np.mean(X)
			mY = np.mean(Y)
			length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
			angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
			polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
			cv2.fillConvexPoly(cur_canvas, polygon, colors[firstlimb_ind])
			canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

	return canvas

import json
def readkeypointsfile_json(myfile):
	'''
	return four list
	'''
	f = open(myfile, 'r')
	json_dict = json.load(f)
	people = json_dict['people']
	posepts =[]
	facepts = []
	r_handpts = []
	l_handpts = []
	for p in people:
		posepts += p['pose_keypoints_2d']
		facepts += p['face_keypoints_2d']
		r_handpts += p['hand_right_keypoints_2d']
		l_handpts += p['hand_left_keypoints_2d']

	return [posepts, facepts, r_handpts, l_handpts]

def scale_resize(curshape, myshape=(1080, 1920, 3), mean_height=0.0):

	if curshape == myshape:
		return None

	x_mult = myshape[0] / float(curshape[0]) # y, vertical
	y_mult = myshape[1] / float(curshape[1]) # x, horizonal

	if x_mult == y_mult:
		# just need to scale
		return x_mult, (0.0, 0.0)
	elif y_mult > x_mult:
		### scale x and center y
		y_new = x_mult * float(curshape[1])
		translate_y = (myshape[1] - y_new) / 2.0
		return x_mult, (translate_y, 0.0)
	### x_mult > y_mult
	### already in landscape mode scale y, center x (rows)

	x_new = y_mult * float(curshape[0])
	translate_x = (myshape[0] - x_new) / 2.0

	return y_mult, (0.0, translate_x)

def fix_scale_coords(points, scale, translate):
	points = np.array(points)
	points[0::3] = scale * points[0::3] + translate[0]
	points[1::3] = scale * points[1::3] + translate[1]
	return list(points)

def tensorGrid(flow):
    # input flow: [b,2,h,w]
    B, _, H, W = flow.size()
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    return grid


def renderhand(handpts, canvas, threshold = 0.05):
	colors = [[230, 53, 40], [231,115,64], [233, 136, 31], [213,160,13],[217, 200, 19], \
		[170, 210, 35], [139, 228, 48], [83, 214, 45], [77, 192, 46], \
		[83, 213, 133], [82, 223, 190], [80, 184, 197], [78, 140, 189], \
		[86, 112, 208], [83, 73, 217], [123,46,183], [189, 102,255], \
		[218, 83, 232], [229, 65, 189], [236, 61, 141], [255, 102, 145]]

	i = 0
	while i < 63:
		confidence = handpts[i+2]
		if confidence > threshold:
			cv2.circle(canvas, (int(handpts[i]), int(handpts[i+1])), 3, tuple(colors[i // 3]), thickness=-1)
		i += 3

	stickwidth = 2
	linearSeq = [range(1, 4+1), range(5, 8+1), range(9, 12+1), range(13, 16+1), range(17, 20+1)]
	for line in linearSeq:
		for step in line:
			if step != line[len(line) - 1]:
				firstlimb_ind = step
				secondlimb_ind = step + 1
			else:
				firstlimb_ind = 0
				secondlimb_ind = line[0]
			if (handpts[3*firstlimb_ind + 2] > threshold) and (handpts[3*secondlimb_ind + 2] > threshold):
				cur_canvas = canvas.copy()
				Y = [handpts[3*firstlimb_ind], handpts[3*secondlimb_ind]]
				X = [handpts[3*firstlimb_ind + 1], handpts[3*secondlimb_ind + 1]]
				mX = np.mean(X)
				mY = np.mean(Y)
				length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
				angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
				polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
				cv2.fillConvexPoly(cur_canvas, polygon, colors[secondlimb_ind])
				canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

	return canvas

def renderface_sparse(facepts, canvas, numkeypoints, disp=False, threshold = 0.2, smalldot = 4):
	# if numkeypoints == 0:
	# 	return renderface(facepts, myshape, canvas, disp, threshold, getave)    
	# assert (numkeypoints > 0)
	if disp:
		color = tuple([255, 255, 255])
	else:
		color = tuple([0, 0, 0])

	avecons = sum(facepts[2:len(facepts):3]) / 70.0
	if avecons < threshold:
		return canvas

	pointlist = [27, 30, 8, 0, 16, 33, 68, 69] #sparse 8 default
	if numkeypoints == 22:
		pointlist = [27, 30, 8, 0, 16, 31, 33, 35, \
					68, 69, 36, 39, 42, 45, 17, 21, 22, 26, 48, 51, 54, 57] #endpoints
	elif numkeypoints == 9:
		pointlist += [62]

	for i in pointlist:
		point = 3*i
		confidence = facepts[point+2]
		if confidence > 0:
			cv2.circle(canvas, (int(facepts[point]), int(facepts[point+1])), smalldot, color, thickness=-1)

	return canvas