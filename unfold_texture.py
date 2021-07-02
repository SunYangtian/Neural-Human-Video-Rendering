#!/usr/bin/python
''' 
this code is to get incomplete texture from a person image and corresponding densepose result
'''
import numpy as np
import cv2
from scipy.interpolate import griddata
import glob
tex_size = 200

def UnfoldImg(img, IUV):
    '''
    return the texture image of each part corresponding to input
    '''
    TextureIm  = np.zeros([24,200,200,3]) # return array
    global TextureCnt

    grid_x, grid_y = np.mgrid[0:200, 0:200] # (200,200), (200,200). coordinate of each grid
    for partID in range(1, 25):
        # print("processing the %d part" % partID)
        y, x = np.where(IUV[:,:,0] == partID) # (N,) (N,) the index of each dim in which value is true
        pixel = img[y, x] # N * 3
        uv = IUV[y, x, 1:3]/255.*199 # N * 2, (u,v) is coordinate
        if len(uv) == 0:
            # print(" ...... no point in this part")
            continue
        uv = uv[:,::-1].astype(np.int32) # change to vu, which is the index
        # TextureIm[partID-1] = griddata(uv, pixel, (grid_x, grid_y), method='nearest') # (200,200,3)
        TextureIm[partID-1][uv[:,0], uv[:,1], :] = pixel
        TextureCnt[partID-1][uv[:,0], uv[:,1], :] += 1

        # print(TextureIm[partID-1][uv[:,::-1].astype(np.int32)]).shape # (200,200,3)

    return TextureIm

def unfold(img_path, IUV_path):
    img = cv2.imread(img_path)[:,:,::-1]/255.
    IUV = cv2.imread(IUV_path)
    TextureIm = UnfoldImg(img, IUV)
    return TextureIm

def visualizeTex(TextureIm, save_path='./texture0617.jpg', do_close=False):
    visTexture = np.zeros((1200, 800, 3)) # (1200, 800, 3)
    for i in range(4): # x coordinate
        for j in range(6): # y coordinate
            visTexture[(200*j):(200*j+200), (200*i):(200*i+200), :] = TextureIm[(6*i+j), ::-1,:,:]
    if do_close:
        kernel = np.ones((5,5),np.uint8)
        R = cv2.morphologyEx(visTexture[:,:,0], cv2.MORPH_CLOSE, kernel)
        G = cv2.morphologyEx(visTexture[:,:,1], cv2.MORPH_CLOSE, kernel)
        B = cv2.morphologyEx(visTexture[:,:,2], cv2.MORPH_CLOSE, kernel)
        visTexture = np.concatenate((R[:,:,np.newaxis],G[:,:,np.newaxis],B[:,:,np.newaxis]), axis=2)
    cv2.imwrite(save_path, visTexture[:,:,::-1]*255)

def TransferTexture(TextureIm, IUV):
    '''
    tex_img [24,200,200,3] + IUV [h,w,3] -> img [h,w,3]
    '''
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        tex = TextureIm[PartInd-1,:,:,:].squeeze() # (200, 200, 3) get texture for each part.
        #####
        R = tex[:,:,0]
        G = tex[:,:,1]
        B = tex[:,:,2]
        ###############
        x,y = np.where(IUV[:,:,0]==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        # r_current_points = R[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        # g_current_points = G[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        # b_current_points = B[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]*255
        r_current_points = R[((v_current_points)/255.*(tex_size-1)).astype(int),(u_current_points/255.*(tex_size-1)).astype(int)]*255
        g_current_points = G[((v_current_points)/255.*(tex_size-1)).astype(int),(u_current_points/255.*(tex_size-1)).astype(int)]*255
        b_current_points = B[((v_current_points)/255.*(tex_size-1)).astype(int),(u_current_points/255.*(tex_size-1)).astype(int)]*255
        ##  Get the RGB values from the texture images.
        R_im[IUV[:,:,0]==PartInd] = r_current_points
        G_im[IUV[:,:,0]==PartInd] = g_current_points
        B_im[IUV[:,:,0]==PartInd] = b_current_points
    generated_image = np.concatenate((R_im[:,:,np.newaxis],G_im[:,:,np.newaxis],B_im[:,:,np.newaxis]), axis=2 ).astype(np.uint8)
    BG_MASK = generated_image==0
    # generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    return generated_image

def wrap(tex_path, IUV_path):
    '''
    warp texture image from tex_path to IUV
    texture image: [1200, 800, 3]
    '''
    Tex_Atlas = cv2.imread(tex_path)[:,:,::-1]/255. # change to RGB
    TextureIm  = np.zeros([24,tex_size,tex_size,3])
    for i in range(4):
        for j in range(6):
            TextureIm[(6*i+j),:,:,:] = Tex_Atlas[(tex_size*j):(tex_size*j+tex_size), (tex_size*i):(tex_size*i+tex_size), :][::-1,:,:] # inverse the y direction
    IUV = cv2.imread(IUV_path)
    return TransferTexture(TextureIm,IUV)

def wrap_v2(tex_path, I_path, U_path, V_path):
    '''
    warp texture image from tex_path to I,U,V
    texture image: [1200, 800, 3]
    '''
    I = cv2.imread(I_path, cv2.IMREAD_UNCHANGED)[:,:,np.newaxis]
    I = (I / 255. * 24).astype(np.uint8)

    U = cv2.imread(U_path, cv2.IMREAD_UNCHANGED)[:,:,np.newaxis]
    V = cv2.imread(V_path, cv2.IMREAD_UNCHANGED)[:,:,np.newaxis] # change (h,w) to (h,w,1)

    IUV = np.concatenate([I,U,V], axis=2) # (h,w,3)

    Tex_Atlas = cv2.imread(tex_path)[:,:,::-1]/255. # change to RGB
    TextureIm  = np.zeros([24,tex_size,tex_size,3])
    for i in range(4):
        for j in range(6):
            TextureIm[(6*i+j),:,:,:] = Tex_Atlas[(tex_size*j):(tex_size*j+tex_size), (tex_size*i):(tex_size*i+tex_size), :][::-1,:,:] # inverse the y direction
    return IUV, TransferTexture(TextureIm,IUV)

def wrap_v3(tex_path, Probs_path, UVs_path):
    import torch
    Probs = np.load(Probs_path) # (25, h ,w)
    Probs = torch.tensor(Probs)

    Tex_Atlas = cv2.imread(tex_path)[:,:,::-1]/255. # change to RGB
    TextureIm  = np.zeros([24,tex_size,tex_size,3])
    for i in range(4):
        for j in range(6):
            TextureIm[(6*i+j),:,:,:] = Tex_Atlas[(tex_size*j):(tex_size*j+tex_size), (tex_size*i):(tex_size*i+tex_size), :][::-1,:,:] # inverse the y direction
    TextureIm = torch.tensor(TextureIm).to(torch.float32)
    
    gen_im = torch.zeros(3, Probs.shape[1], Probs.shape[2]) # (3, h, w)

    UVs = np.load(UVs_path)
    UVs = torch.tensor(UVs) # (48,h,w)

    for partID in range(1,25):
        texture = TextureIm[(partID-1),:,:].permute(2,0,1) # [3,tex_size,tex_size]
        uv = UVs[(partID-1)*2:partID*2,:,:].permute(1,2,0) # [h,w,2]
        img = torch.nn.functional.grid_sample(texture.unsqueeze(0), uv.unsqueeze(0)) # [1,3,h,w]
        prob = Probs[partID,:,:].unsqueeze(0) # [1,h,w]
        gen_im += img[0] * prob # [bs,3,h,w]

    return gen_im.permute(1,2,0).cpu().numpy() * 255

if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    IUV_dir = sys.argv[2]

    img_paths = sorted(glob.glob(img_dir+'/*.jpg'))
    IUV_paths = sorted(glob.glob(IUV_dir+'/*.png'))
    # if len(IUV_paths) == 0:
    #     IUV_paths = sorted(glob.glob(IUV_dir+'/*.jpg'))
    
    assert(len(img_paths) == len(IUV_paths)), "img_paths: %d, IUV_paths: %d " % (len(img_paths),len(IUV_paths))

    ## for generation
    if True:
        IMAGE_START = 0
        IMAGE_NUM = len(IUV_paths)
        IMAGE_NUM = 5000
        print("total images: %d" % IMAGE_NUM)

        TextureCnt = np.ones([24,200,200,3], dtype=np.int32)
        TextureIm_  = np.zeros([24,200,200,3])
        for idx, (img_path,IUV_path) in enumerate(zip(img_paths,IUV_paths)):
            if idx < IMAGE_START:
                continue
            if idx >= IMAGE_NUM + IMAGE_START:
                break
            TextureIm_ += unfold(img_path, IUV_path)
            print("processing the %d image" % (idx+1))
        TextureCnt[TextureCnt > 1] -= 1
        TextureIm_ = TextureIm_ / TextureCnt
        # visualizeTex(TextureIm_, '/home/sunyangtian/104/iPER/iPER_1024_label/007/3/texture.jpg')
        # visualizeTex(TextureIm_, '/home/sunyangtian/104/new_data/dance16/texture.jpg')
        visualizeTex(TextureIm_, '/home/sunyangtian/104mnt/DanceDataset/dance14/texture.jpg')


    ### for single unfold test
    if False:
        TextureIm_  = np.zeros([24,200,200,3])
        TextureCnt = np.ones([24,200,200,3], dtype=np.int32)
        IUV_path = IUV_paths[0]
        img_path = img_paths[0]
        TextureIm_ = unfold(img_path, IUV_path)
        visualizeTex(TextureIm_, '/home/sunyangtian/104/iPER/iPER_1024_label/001/12/single_texture.jpg', do_close=True)


    ### for wrap test
    if False:
        tex_dir = '/home/sunyangtian/104/iPER/iPER_1024_label/001/12/part_texture/1'
        tex_paths = sorted(glob.glob(tex_dir+'/*.jpg'))
        # IUV_path = IUV_paths[530]
        IUV_path = '/home/sunyangtian/104/iPER/iPER_1024_label/008/3/densepose/2/frame00092_IUV.png'
        # tex_path = tex_paths[530]
        tex_path = '/home/sunyangtian/104/iPER/iPER_1024_label/001/12/test_texture.jpg'
        # tex_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_static/checkpoints/static_train/web/images/epoch087_synthesized_texture.jpg'
        # tex_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_synthesized_texture.jpg'
        visTexture = wrap(tex_path, IUV_path)
        save_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/frame00092_wrap_initial.jpg'
        # print("test:", visTexture[190:193,550:553])
        cv2.imwrite(save_path, visTexture[:,:,::-1])
    
    ### for wrap_v2 test
    if False:
        tex_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_static/checkpoints/static_train/web/images/epoch087_synthesized_texture.jpg'
        # tex_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_synthesized_texture.jpg'
        # tex_path = '/home/sunyangtian/104/iPER/iPER_1024_label/001/12/texture.jpg'
        I_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_Probs.jpg'
        U_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_U.jpg'
        V_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_V.jpg'

        IUV, wraped_Im = wrap_v2(tex_path, I_path, U_path, V_path)
        save_dense_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/frame00092_gen_densepose.jpg'
        # cv2.imwrite(save_dense_path, IUV)
        save_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/frame00092_wrap_v2_static.jpg'
        cv2.imwrite(save_path, wraped_Im[:,:,::-1])


    ### for wrap_v3 test
    if False:
        # tex_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_static/checkpoints/static_train/web/images/epoch087_synthesized_texture.jpg'
        # tex_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_synthesized_texture.jpg'
        tex_path = '/home/sunyangtian/104/iPER/iPER_1024_label/001/12/texture.jpg'
        Probs_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_Probs.npy'
        UVs_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/images/frame00092_keypoints_UVs.npy'

        wraped_Im = wrap_v3(tex_path, Probs_path, UVs_path)
        save_dense_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/frame00092_gen_densepose.jpg'
        # cv2.imwrite(save_dense_path, IUV)
        save_path = '/home/sunyangtian/104/code/pix2pixHD_avatar_08/test_single_input/0809_train_2/test_latest/frame00092_wrap_v3_initial.jpg'
        cv2.imwrite(save_path, wraped_Im[:,:,::-1])

    # IUV_path = './demo_data/frame00003_IUV.png'
    # IUV = cv2.imread(IUV_path)
    # visTexture = TransferTexture(TextureIm_, IUV)
