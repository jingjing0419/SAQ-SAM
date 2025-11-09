import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader
# from DataLoader import TestingDataset
from ptq4sam.repq_utils import MatMul
import random

from .quant_modules_rep import QuantConv2d, QuantLinear, QuantMatMul
# from test_quant import to_device
# from utils import train_transforms, get_boxes_from_mask, init_point_sampling, train_transforms2
import cv2
from torch.nn import Parameter
from PIL import Image
import numpy as np


def calib_gausiion_noize(args, model):
    calib_data=torch.randn((args.calib_batch_size, 1, args.image_size, args.image_size)).cuda()
    # calib_data = calib_data.repeat(1,3,1,1)
    # calib_data=(calib_data-calib_data.min())/(calib_data.max()-calib_data.min()+1e-10)*255
    # calib_data = torch.floor(calib_data)
    print('calib...')
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)

def calib_gausiion_noize_3C(args, model):
    calib_data=torch.randn((args.calib_batch_size, 3, args.image_size, args.image_size)).cuda()
    print('calib...')
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)
        
def calib_train_data(args, model):
    calib_data = torch.load('calib_data/calib_data_random_train.pth')
    calib_data = calib_data.float().cuda()
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)
        
        
        
def calib_FLARE_b1(args, model):
    calib_data = torch.load('calib_data/calib_data_FLARE.pth')
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)
        
        
def calib_distilled(args, q_model):
    calib_data = torch.load(args.distill_data_path)
    calib_data = calib_data.float().cuda()
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(calib_data)
        
def calib_distilled2(args, q_model):
    calib_data = torch.load(args.distill_data_path)
    calib_data = calib_data.float().cuda()
    if calib_data.shape[1]==1:
        calib_data = calib_data.repeat(1,3,1,1)
        pixel_mean = torch.tensor([123.675/255, 116.28/255, 103.53/255]).view(1,3,1,1).cuda()
        pixel_std = torch.tensor([58.395/255, 57.12/255, 57.375/255]).view(1,3,1,1).cuda()
        calib_data = (calib_data - pixel_mean)/pixel_std
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(calib_data)

def calib_load_data(args, q_model, load_calib_data_path):
    calib_data = torch.load(load_calib_data_path)
    print('loaded from', load_calib_data_path)
    calib_data = calib_data.float().cuda()
    if calib_data.shape[1]==1:
        calib_data = calib_data.repeat(1,3,1,1)
        pixel_mean = torch.tensor([123.675/255, 116.28/255, 103.53/255]).view(1,3,1,1).cuda()
        pixel_std = torch.tensor([58.395/255, 57.12/255, 57.375/255]).view(1,3,1,1).cuda()
        calib_data = (calib_data - pixel_mean)/pixel_std
        print('norm the data')
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(calib_data)

def calib_load_data_gc(args, q_model, calib_pth_gray, calib_pth_color):
    data_gray = torch.load(calib_pth_gray)
    data_gray = data_gray.repeat(1,3,1,1)
    data_color = torch.load(calib_pth_color)
    # print(data_gray.shape)
    # print(data_color.shape)
    # exit()
    data_calib = torch.cat((data_gray, data_color), dim=0).cuda()
    print(data_calib.shape)
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(data_calib)

def quant_image_encoder(args, model, input_quant_params={}, weight_quant_params={}):
    # input
    input_quant_params_embed = deepcopy(input_quant_params)
    input_quant_params_embed['n_bits'] = 4

    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    # if args.log2_quant:
    input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    module_dict={}
    for name, m in model.predictor.model.image_encoder.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            if 'embed' in name:
                # continue
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params_embed,
                    weight_quant_params
                )
            else:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params,
                    weight_quant_params
                )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'blocks' in name and ('Adapter' not in name):
                if 'qkv' in name or 'lin1' in name:
                    new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, True) 
                else:   
                    new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, True)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, False)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model

def quant_mask_decoder(args, model, input_quant_params={}, weight_quant_params={}):
    # input
    input_quant_params_embed = deepcopy(input_quant_params)
    input_quant_params_embed['n_bits'] = 8

    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    # if args.log2_quant:
    # input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    module_dict={}
    for name, m in model.predictor.model.mask_decoder.transformer.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        # if 'cross_attn_token_to_image' in name:
        #     continue
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantConv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                input_quant_params,
                weight_quant_params
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            print(name)
            idx = idx + 1 if idx != 0 else idx
            if ('cross_attn_token_to_image' in name ) and ('k_proj' in name or 'v_proj' in name):
                pass
                # new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, False) 
            elif ('cross_attn_image_to_token' in name ) and ('q_proj' in name):
                pass
            else:   
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, False)
            
                new_m.weight.data = m.weight.data
                new_m.bias = m.bias
                setattr(father_module, name[idx:], new_m)
                
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx

            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model




def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
            
def scale_reparameter(args, q_model):
    with torch.no_grad():
        module_dict={}
        q_model_slice = q_model.image_encoder.blocks
        for name, module in q_model_slice.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            # if 'norm1' in name or 'norm2' in name or 'norm' in name:
            if 'norm1' in name or 'norm2' in name:
                # print('norm')
                # print(name)
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                    # print('father_module.attn.qkv')
                    # print(next_module)
                elif 'norm2' in name:
                    next_module = father_module.mlp.lin1
                    # print('father_module.mlp.lin1')
                    # print(next_module)
  
                
                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b
                # print(next_module.weight.data.shape)        
                # exit()
                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False


def scale_reparameter2(args, q_model):
    global_atten_name=['2.norm1','5.norm1','8.norm1','11.norm1']
    with torch.no_grad():
        module_dict={}
        q_model_slice = q_model.image_encoder.blocks
        for name, module in q_model_slice.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            # if 'norm1' in name or 'norm2' in name or 'norm' in name:
            if 'norm1' in name or 'norm2' in name:
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.lin1
  
                if 'norm2' in name:
                    act_delta = next_module.input_quantizer.delta.reshape(-1)
                    act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                    act_min = -act_zero_point * act_delta
                    
                    target_delta = torch.mean(act_delta)
                    target_zero_point = torch.mean(act_zero_point)
                    target_min = -target_zero_point * target_delta

                    r = act_delta / target_delta
                    b = act_min / r - target_min

                    module.weight.data = module.weight.data / r
                    module.bias.data = module.bias.data / r - b
                    next_module.weight.data = next_module.weight.data * r
                    if next_module.bias is not None:
                        next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                    else:
                        next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                        next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                    next_module.input_quantizer.delta = target_delta
                    next_module.input_quantizer.zero_point = target_zero_point
                    next_module.input_quantizer.channel_wise = False
                    next_module.weight_quantizer.inited = False

def scale_reparameter3(args, q_model):
    with torch.no_grad():
        module_dict={}
        q_model_slice = q_model.image_encoder.blocks
        for name, module in q_model_slice.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            # if 'norm1' in name or 'norm2' in name or 'norm' in name:
            if 'norm1' in name or 'norm2' in name:
                q_adapter = None
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.lin1
                    if args.encoder_adapter:
                        q_adapter = father_module.Adapter.channel[0]
                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b
                # print(next_module.weight.data.shape)        
                # exit()
                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False

                if q_adapter != None:
                    print('adapter')
                    q_adapter.weight_quantizer.inited = False
                    q_adapter.input_quantizer.inited = False