import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_Avatar import Pix2PixHD_Avatar, InferenceModel
        # from .pix2pixHD_Avatar_TexG import Pix2PixHD_Avatar, InferenceModel
        if opt.isTrain:
            model = Pix2PixHD_Avatar()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
