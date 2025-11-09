# Copyright (c) OpenMMLab. All rights reserved.
# 基于V10，整理代码
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import pandas as pd
from py import log
from sympy import im
import torch
from mmcv import Config, DictAction
from mmcv.utils import get_logger
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
import sys
sys.path.append("/home/zhangjing/PTQ4SAM_Proj_A2/SAQ-SAM")
import copy
import logging
import torch.nn as nn
import torch.distributed as dist
from ptq4sam.model.quant_model import specials
from ptq4sam.model.quant_model import bimodal_adjust, enable_rank_observe, enable_get_ori_ATM, QuantDecoderOurAttentionBlock
from ptq4sam.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all, enable_calibration_with_quantization, enable_quantization_plus, disable_all_observer
from ptq4sam.quantization.quantized_module import QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul, QuantizedTransformerLayer, QuantizedTransformerStage
from ptq4sam.quantization.fake_quant import QuantizeBase
from ptq4sam.quantization.observer import ObserverBase, AvgMSEObserver, AvgMinMaxObserver
from recon import reconstruction_blk, reconstruction_IE, reconstruction_MD, save_sam_inputs
import utils
import random
import numpy as np

from torch import Tensor
from typing import Tuple
import wandb


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


class Auxiliary_pack(nn.Module):
    # Package the components separated from the original model.
    def __init__(self):
        super(Auxiliary_pack, self).__init__()
        self.components = nn.ModuleDict()
        
        
    def add_copy_module(self, base_model, copy_components):
        for name, module in base_model.named_modules():
            if name == copy_components:
                n = name.split('.')[-1]
                self.components[n] = module



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config',
                        default='./projects/configs/yolox/yolo_l-sam-vit-b.py', 
                        help='test config file path')
    parser.add_argument(
        '--work-dir',
        default='result/tmp',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='segm',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', default=False, type=bool, help='show results')
    parser.add_argument(
        '--show-dir', default='/home/zhangjing/PTQ4SAM_Proj_A2/hahaha/result_3layer_encoder',help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        default=True,
        help='whether to use gpu to collect results.')
    
    parser.add_argument(
        '--quant-encoder',
        action='store_true', 
        default=False,
        help='whether to quant encoder.')
    
    parser.add_argument(
        '--quant-decoder',
        action='store_true', 
        default=False,
        help='whether to quant encoder.')
    
    parser.add_argument(
        '--fp',
        action='store_true',
        help='fp')
    
    parser.add_argument(
        '--test_fp',
        action='store_true',
        help='fp')

    parser.add_argument(
        '--save_sam_path',
        type=str,
        default='/mnt/data/zhangjing/PTQ4SAM_Proj_A2/hahaha/save_sam_ckpt/',
        help='save_sam_path')
    
    parser.add_argument(
        '--recon_from_load',
        action='store_true',
        help='')
    
    parser.add_argument(
        '--test_load_model',
        action='store_true',
        help='')

    parser.add_argument(
        '--load_sam_path',
        type=str,
        default='',
        help='load_sam_path')
    
    parser.add_argument(
        '--short4cut',
        action='store_true',
        help='short cut for G mem')
    
    parser.add_argument(
        '--brecq',
        action='store_true',
        help='Brecq')
    
    parser.add_argument(
        '--resign_end',
        action='store_true',
        help='resign_end')

    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--q_config',
        type=str,
        default='./exp/config_SA_66.yaml',
        help='quantization config files')
    
    parser.add_argument(
        '--quant_act_list',
        default=None, 
        nargs='+', 
        type=str)
    
    parser.add_argument(
        '--FFC',
        default=False,  
        action='store_true', 
        help='use FFC')
    
    parser.add_argument(
        '--recon_encoder',
        default='CAMS',  
        type=str,
        help='no, original, CAMS')
    
    parser.add_argument(
        '--recon_decoder',
        default='CLM',  
        type=str,
        help='no, original, CLM')
    
    parser.add_argument(
        '--calib_num',
        default=2,  
        type=int,
        help='no, original, CLM')
    
    parser.add_argument(
        '--test_notes',
        default='',  
        type=str,
        help='no, original, CLM')
    
    parser.add_argument(
        '--CAM_loss',
        default='PAR',  
        type=str,
        help='PAR,ori')
    
    parser.add_argument(
        '--alpha',
        default=0.5,  
        type=float,
        help='Hyperparameter of loss function')
    
    parser.add_argument(
        '--beta',
        default=0,  
        type=float,
        help='Hyperparameter of loss function')
    
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        default=False)
    
    parser.add_argument(
        '--gpu2',
        action='store_true',
        default=False)
    
    parser.add_argument(
        '--recon_unit',
        type=str,
        default='per_stage',
        help='per_stage, per_layer')
    
    parser.add_argument(
        '--save_sam_calib',
        default=False,
        action='store_true',
        help='')
    
    parser.add_argument(
        '--save_sam_recon',
        default=False,
        action='store_true',
        help='')
    
    parser.add_argument(
        '--FFC_threshold',
        default=0.5,  
        type=float
        )
    
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

class activation_hook:
    def __init__(self, module, out_act = True):
        if out_act:
            self.hook = module.register_forward_hook(self.hook_fn_output)
        else:
            self.hook = module.register_forward_hook(self.hook_fn_input)
            
    def hook_fn_output(self, module, input, output):
        self.feature = output
    def hook_fn_input(self, module, input, output):
        print('input')
        self.feature = input

    def remove(self):
        self.hook.remove()
        
def main():
    set_seed(1234)
    
    global logger
    args = parse_args()
    brecq = args.brecq
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    
    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)
   

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)
                plg_lib = importlib.import_module(_module_path)


    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif (cfg.model.get('backbone', None) is not None
          and 'init_cfg' in cfg.model.backbone):
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    if args.q_config:
        q_config = utils.parse_config(args.q_config)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(args.work_dir, f'{timestamp}.log')
        logger = get_logger(name='ptq4sam', log_file=log_file, log_level=logging.INFO)
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')
        
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    cali_data = utils.load_calibration(cfg, distributed, q_config.calibrate)


    # build the model and load checkpoint
    cfg.model.train_cfg = None

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # cfg.get('test_cfg')=None

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = {}
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    
    if not args.fp:
        if not args.test_load_model:
            # PTQ4SAM configuration
            logger.info('quant encoder')
            logger.info(args.quant_encoder)
            logger.info('do sign')
            logger.info(q_config.ptq4sam.BIG)
            logger.info('do log_quant')
            logger.info(q_config.ptq4sam.AGQ)
            q_config.ptq4sam.threshold=args.FFC_threshold
            logger.info('q_config.ptq4sam.threshold:{}'.format(q_config.ptq4sam.threshold))
           

            assert args.q_config is not None
            for para in model.parameters():
                para.requires_grad = False
            model = quantize_model(model, q_config, args)
            model.cuda()
            model.eval()
            
            # print(model)
            if args.use_wandb:
                try:
                    # os.environ["WANDB_MODE"] = "offline"
                    wandb.init(
                    # set the wandb project where this run will be logged
                        project="exp",
                        # track hyperparameters and run metadata
                        config={
                        "scale_lr:": 0.01,
                        "iters": 500,
                        }
                    )
                except Exception as e:
                    logging.error(f"WandB 连接失败: {e}. 将继续运行程序。")

        
            if args.test_fp:
                model.det_model.cuda()
                
                if not distributed:
                    if args.show_dir is not None and 'gt' in args.show_dir:
                        gt = True
                    else:
                        gt = False
                    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
                    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                            args.show_score_thr, gt=gt)
                else:
                    model = build_ddp(
                        model,
                        cfg.device,
                        device_ids=[int(os.environ['LOCAL_RANK'])],
                        broadcast_buffers=False)

                    # In multi_gpu_test, if tmpdir is None, some tesnors
                    # will init on cuda by default, and no device choice supported.
                    # Init a tmpdir to avoid error on npu here.
                    if cfg.device == 'npu' and args.tmpdir is None:
                        args.tmpdir = './npu_tmpdir'

                    outputs = multi_gpu_test(
                        model, data_loader, args.tmpdir, args.gpu_collect
                        or cfg.evaluation.get('gpu_collect', False))

                rank, _ = get_dist_info()
                if rank == 0:
                    if args.out:
                        print(f'\nwriting results to {args.out}')
                        mmcv.dump(outputs, args.out)
                    kwargs = {} if args.eval_options is None else args.eval_options
                    if args.format_only:
                        dataset.format_results(outputs, **kwargs)
                    if args.eval:
                        eval_kwargs = cfg.get('evaluation', {}).copy()
                        # hard-code way to remove EvalHook args
                        for key in [
                                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                                'rule', 'dynamic_intervals'
                        ]:
                            eval_kwargs.pop(key, None)
                        eval_kwargs.update(dict(metric=args.eval, **kwargs))
                        metric = dataset.evaluate(outputs, **eval_kwargs)
                        print(metric)
                        logger.info(q_config)
                        logger.info(metric)
                        metric_dict = dict(config=args.config, metric=metric)
                        if args.work_dir is not None and rank == 0:
                            mmcv.dump(metric_dict, json_file)
                exit()
            #############
            
        
            fp_model = copy.deepcopy(model)
            # 对于huge和large模型，使用两个GPU重建
            if args.gpu2:
                model_occupy = copy.deepcopy(fp_model.predictor.model.mask_decoder)
                model_occupy.cuda(1)

            if not args.recon_from_load:
                calibrate(args, model, cali_data, q_config.ptq4sam.BIG)
                
                enable_quantization(model)
                if args.save_sam_calib:
                    if len(args.save_sam_path) > 0:
                        save_pth_path = args.save_sam_path+timestamp+'_calib.pth'
                        try:
                            logger.info('save sam:')
                            logger.info(save_pth_path)
                            torch.save(model.predictor.model, save_pth_path)
                        except:
                            logger.info('err no save')
            else:
                load_path = args.load_sam_path
                logger.info('load sam:')
                logger.info(load_path)
                model.predictor.model = torch.load(load_path)
                enable_quantization(model)
            
            # Add auxiliary modules required for PAR interaction
            Aux_module = Auxiliary_pack()
            Aux_module.add_copy_module(fp_model, 'predictor.model.image_encoder.neck')
            Aux_module.add_copy_module(fp_model, 'predictor.model.mask_decoder')
            
            logger.info('decoder reconstruction supervised by FP model')
            if hasattr(q_config, 'recon'):
                logger.info('begin to do reconstruction')
                recon_model(args, model, fp_model, cali_data, q_config.recon, Aux_module, brecq=brecq)
            
            if args.gpu2:
                del model_occupy
                
            disable_all(model)
            if args.quant_encoder:
                enable_quantization(model.predictor.model.image_encoder)
            if args.quant_decoder:
                enable_quantization(model.predictor.model.mask_decoder)
            
            
            logger.info('quantized model')
            logger.info(model.predictor.model)

            
            if args.save_sam_recon:
                if len(args.save_sam_path) > 0:
                    save_pth_path = args.save_sam_path+timestamp+'_recon.pth'
                    try:
                        # save_path = args.save_sam_path
                        logger.info('save sam:')
                        logger.info(save_pth_path)
                        torch.save(model.predictor.model, save_pth_path)
                    except:
                        logger.info('err no save')
        else:
            load_path = args.load_sam_path
            logger.info('load sam:')
            logger.info(load_path)
            model.predictor.model = torch.load(load_path)
            disable_all(model)
            if args.quant_encoder:
                enable_quantization(model.predictor.model.image_encoder)
            if args.quant_decoder:
                enable_quantization(model.predictor.model.mask_decoder)
        
    model.det_model.cuda()
    torch.cuda.empty_cache()
    
    
    # ---------------------------------------
    # Note: It was found in [1] that the implementation of PTQ4SAM has some bugs in the drop_prob settings, 
    # and we used PTQ4SAM as our baseline without fixing this issue. 
    # To reproduce the results in the paper, please keep the following code commented out. 
    # If you wish to perform a fair comparison, uncommenting the code can fix the bug. 
    # Note that this bug only exists when reconstruction is enabled, so experimental results without the reconstruction step are not affected. 
    # We will update the repository with the experimental results after fixing the bug as soon as possible.    
    # [1] AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model
    
    # for n, m in model.named_modules():
    #     if hasattr(m, 'drop_prob'):
    #         m.drop_prob = 1
    # ---------------------------------------
    
    if not distributed:
        if args.show_dir is not None and 'gt' in args.show_dir:
            gt = True
        else:
            gt = False
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr, gt=gt)
    else:
        
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        torch.cuda.empty_cache()
        # In multi_gpu_test, if tmpdir is None, some tesnors
        # will init on cuda by default, and no device choice supported.
        # Init a tmpdir to avoid error on npu here.
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            logger.info(q_config)
            logger.info(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)

def quantize_model(model, config_quant, args):
    # , config_quant.ptq4sam
    def replace_module(module, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True):
        for name, child_module in module.named_children():
            if 'patch_embed' in name or 'output_upscaling' in name or 'iou_prediction_head' in name or 'output_hypernetworks_mlps' in name:
                continue
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, ptq4sam_config))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.GELU)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, nn.Identity())
                else:
                    pass
            elif isinstance(child_module, nn.Identity):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, ptq4sam_config, qoutput)
            
    logger.info('**********quant whole model')
    model.predictor.model.image_encoder = specials[type(model.predictor.model.image_encoder)](model.predictor.model.image_encoder, config_quant.encoder_w_qconfig, config_quant.encoder_a_qconfig, config_quant.ptq4sam)

    replace_module(model.predictor.model.mask_decoder, config_quant.decoder_w_qconfig, config_quant.decoder_a_qconfig, config_quant.ptq4sam)
    'set first layer\'s weight to 8-bit'
   
    
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
            
    return model


@torch.no_grad()
def calibrate(args, model, cali_data, BIG):
    st = time.time()
    # Bimodal detection and merge (BIG) of PTQ4SAM
    model.extract_feat(cali_data[0])
    if BIG:
        bimodal_adjust(model, logger=logger)
    
    logger.info('********quant activation ***********')
    # PCC calibration for QK activation
    # calib 1 epoch with 2 iterations, for adapting large-scale value change.
    if args.FFC:
        for j in range(1):
            logger.info('**********QK calib epoch {}'.format(j))
            enable_get_ori_ATM(model, logger)
            # get FP attention focus mask and store it in module
            model.extract_feat(cali_data[j])
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for i in range(2):
                # begin PCC calibration
                logger.info('**********QK calib epoch {} itr {}'.format(j, i))
                enable_rank_observe(model, logger)
                model.extract_feat(cali_data[j])
            # delete cached ATM
            for name, submodule in model.named_modules():
                if isinstance(submodule, QuantDecoderOurAttentionBlock):
                    submodule.ori_ATM = None
    else:
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    
    for i in range(args.calib_num):
        logger.info('######callib_{}###########'.format(i))
        model.extract_feat(cali_data[i])
        
    rank, world_size = get_dist_info()
    if world_size!=1:
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                module.min_val.data /= world_size
                module.max_val.data /= world_size
                dist.all_reduce(module.min_val.data)
                dist.all_reduce(module.max_val.data)
    
    enable_calibration_with_quantization(model, quantizer_type='weight_fake_quant')  
    # enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')  
    model.extract_feat(cali_data[0])
    
    disable_all_observer(model)
        
    ed = time.time()
    if rank == 0:
        logger.info('the calibration time is {}'.format(ed - st))

cnt_tmp = 0
def recon_model(args, model, fp_model, cali_data, recon_config, Aux_module, brecq=False):
    global cnt_tmp
    if args.recon_unit == 'per_block':
        recon_module_pack = (QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)
    elif args.recon_unit == 'per_layer':
        recon_module_pack = (QuantizedTransformerLayer, QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)
    elif args.recon_unit == 'per_stage':
        recon_module_pack = (QuantizedTransformerStage, QuantizedTransformerLayer, QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)
    else:
        raise NotImplementedError('check recon_unit setting')
    
    if len(cali_data) > 16:
        recon_config.keep_gpu = False
    
    disable_all(model)
    
    
    
    def _recon_model_IE(args, module, fp_module, save_sam_data):
        global cnt_tmp
        # fp_model = copy.deepcopy(model)
        for name, child_module in module.named_children():
            if isinstance(child_module, recon_module_pack):
                enable_quantization(child_module)    
                cnt_tmp+=1
                logger.info('begin reconstruction for module:{}'.format(name))
                reconstruction_IE(args, model, fp_model, child_module, getattr(fp_module, name), cali_data, save_sam_data, recon_config, Aux_module, name)
                setattr(fp_module, name, copy.deepcopy(child_module))
            else:
                _recon_model_IE(args, child_module, getattr(fp_module, name), save_sam_data)
    
    
    def _recon_model_MD(args, module, fp_module, save_sam_data):
        for name, child_module in module.named_children():
            
            if isinstance(child_module, recon_module_pack):
                enable_quantization(child_module)
                logger.info('begin reconstruction for module:{}'.format(name))
                # logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                reconstruction_MD(args, model, fp_model, child_module, getattr(fp_module, name), cali_data, save_sam_data, recon_config, Aux_module, name)
                # setattr(fp_module, name, child_module)
            else:
                _recon_model_MD(args, child_module, getattr(fp_module, name), save_sam_data)
    
    def _recon_model_blk(args, module, fp_module):
        global cnt_tmp
        
        assert args.recon_unit == 'per_block'
        
        for name, child_module in module.named_children():
            if isinstance(child_module, recon_module_pack):
                cnt_tmp+=1
                logger.info('begin reconstruction for module:{}'.format(name))
                # logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                reconstruction_blk(args, model, fp_model, child_module, getattr(fp_module, name), cali_data, save_sam_data, recon_config, name)
            else:
                _recon_model_blk(args, child_module, getattr(fp_module, name))
    
    image_data, image_pes, sparse_prompt_embeds, dense_prompt_embeds = save_sam_inputs(fp_model, cali_data, max_pt = 16)
    save_sam_data = {'image_data':image_data, 
                    'image_pes':image_pes, 
                    'sparse_prompt_embeds':sparse_prompt_embeds, 
                    'dense_prompt_embeds':dense_prompt_embeds}
    
    # Start reconstruction
    st = time.time()
    if args.recon_encoder == 'original':
        if brecq:
            enable_quantization(model,'weight_fake_quant')
        else:
            enable_quantization(model)
        _recon_model_blk(args, model.predictor.model.image_encoder, fp_model.predictor.model.image_encoder)
    elif args.recon_encoder == 'CAMS':
        _recon_model_IE(args, model.predictor.model.image_encoder, fp_model.predictor.model.image_encoder, save_sam_data)
    else:
        pass
    
    mid=time.time()
    
    if args.recon_decoder == 'original':
        # recon_config.iters = 20000
        
        _recon_model_blk(args, model.predictor.model.mask_decoder, fp_model.predictor.model.mask_decoder)
    elif args.recon_decoder == 'CLM':
        torch.cuda.empty_cache()

        _recon_model_MD(args, model.predictor.model.mask_decoder, fp_model.predictor.model.mask_decoder, save_sam_data)
    else:
        pass
    
    ed=time.time()
    logger.info('[REC Time:] image_encoder:{}, mask_decoder:{}, total:{}'.format(mid-st, ed-mid, ed-st))
    

def recon_model_ori(model, fp_model, cali_data, recon_config,brecq=False):
    if len(cali_data) > 16:
        recon_config.keep_gpu = False
    
    if brecq:
        enable_quantization(model,'weight_fake_quant')
    else:
        enable_quantization(model)
    
    def _recon_model(module, fp_module):
        for name, child_module in module.named_children():
            
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)):
                logger.info('begin reconstruction for module:{}'.format(name))
                # logger.info('begin reconstruction for module:\n{}'.format(str(child_module)))
                reconstruction_blk(model, fp_model, child_module, getattr(fp_module, name), cali_data, recon_config)
            else:
                _recon_model(child_module, getattr(fp_module, name))
    # Start reconstruction
    _recon_model(model, fp_model)
    # _recon_model(model.predictor.model.mask_decoder, fp_model.predictor.model.mask_decoder)
    
        


if __name__ == '__main__':
    main()
