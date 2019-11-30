import os
import sys
import torch

def fcn_opts(dataset):
    opt = type('', (), {})()
    
    opt.data_dir = sys.path[0]+'/../../data/'
    opt.task = 'ctdet'
    split = 'train'
    opt.num_workers = 4 # Number of dataloader threads (Default: 4)
    opt.not_cuda_benchmark = True
    opt.seed = 0
    opt.batch_size = 1
    opt.lr = 1.25e-4 # default=1.25e-4
    opt.num_classes = dataset.num_classes
    opt.cat_spec_wh = False
    opt.arch = 'res_18' # Default: dla_34
    opt.head_conv = 64 # '64 for resnets and 256 for dla.'
    opt.mse_loss = True
    opt.dense_wh = True # apply weighted regression near center or just apply 
    opt.gpus = [0];
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.device = device
    start_epoch = 0
    opt.num_epochs = 1
    opt.exp_id = 'exp0' # default: 'default'
    opt.hide_data_time = True
    opt.print_iter = 1
    opt.debug = 0 # Default 0
    opt.test = False
    opt.keep_res = False

    opt.lr_step = '90,120'
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = '1'
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.not_reg_offset = False
    opt.reg_offset = not opt.not_reg_offset

    opt.fix_res = not opt.keep_res

    opt.head_conv = -1
    if opt.head_conv == -1: # init default head_conv
        opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    opt.trainval = True
    if opt.trainval:
        opt.val_intervals = 100000000

    opt.master_batch_size = -1

    if opt.debug > 0: 
        opt.num_workers = 0
        opt.batch_size = 1
        opt.gpus = [opt.gpus[0]]
        opt.master_batch_size = -1

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)

    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)

    opt.root_dir = sys.path[0]+'/..'+'/../'
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')

    opt.resume = True
    opt.load_model = ''

    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')

    opt.not_rand_crop = True
    opt.shift = 0.1
    opt.scale = 0.4
    opt.rotate = 0
    opt.flip = 0.5
    opt.no_color_aug = True

    opt.down_ratio = 4
    opt.hm_gauss = 4 # 4 If resolution is (512,512)

    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    opt.input_res = -1
    opt.input_h = -1
    opt.input_w = -1

    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)

    opt.heads = {'hm': opt.num_classes,
                 'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
    if opt.reg_offset:
        opt.heads.update({'reg': 2})

    opt.reg_loss = 'l1'
    opt.hm_weight = 1
    opt.off_weight = 1
    opt.wh_weight = 0.1

    opt.norm_wh = True

    opt.eval_oracle_hm = True
    opt.eval_oracle_wh = False
    opt.eval_oracle_offset = False
    return opt
