import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader, CreateDataLoader_new
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

if opt.test_val:
    data_loader = CreateDataLoader_new(opt)
else:
    data_loader = CreateDataLoader(opt)

dataset = data_loader.load_data_test()
visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name+"_fg", '%s_%s' % (opt.phase, opt.which_epoch))
web_dir = os.path.join(opt.results_dir, opt.name+"", '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if i >= 500:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        # fg_image, bg_image, bg_mask, fake_image, gen_texture, UVs, Probs, mask_tex = model.inference(data['texture'], data['Pose'], data['pose'], data['bg'])
        fg_image, bg_image, bg_mask, fake_image, gen_texture, UVs, Probs, mask_tex = model.inference(data['texture'], data['Pose'], data['bg'])
        
    # visuals = OrderedDict([('input_label', util.tensor2im(data['Pose'][0])),
    #                        ('synthesized_image', util.tensor2im(generated.data[0]))])
    # visuals = OrderedDict([('input_label', util.tensor2im(data['Pose'][0])), ('synthesized_image', util.tensor2im(fake_image.data[0]))])
    visuals = OrderedDict([('synthesized_image', util.tensor2im(fake_image.data[0]))])
    img_path = data.get('path', ['frame%08d.jpg' % i])
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
