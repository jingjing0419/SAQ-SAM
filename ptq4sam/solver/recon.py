import numpy as np
import torch
import torch.nn as nn
import logging
import time
from utils import DataSaverHook, StopForwardException, DataSaverPreHook
from ptq4sam.quantization.quantized_module import QuantizedModule, QuantLinear
from ptq4sam.quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase , AdaptiveGranularityQuantize
from typing import Optional, Tuple, Type
import torch.nn.functional as F
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import window_unpartition
from ptq4sam.model.quant_model import AttentionOverlapLoss
import wandb
import copy
import sys
sys.path.append("/home/zhangjing/PTQ4SAM_Proj_A2/SAQ-SAM")

from ptq4sam.quantization.quantized_module import QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul, QuantizedTransformerLayer, QuantizedTransformerStage
from ptq4sam.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all, enable_calibration_with_quantization, enable_quantization_plus, disable_all_observer, disable_quantization
import gc
import os
import torch.distributed as dist
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)


logger = logging.getLogger('ptq4sam')

def save_inp_oup_data_MD(model, module, save_sam_data, store_inp=False, store_oup=False):
    device = next(module.parameters()).device
    mask_decoder = model.predictor.model.mask_decoder
    image_pes = save_sam_data['image_pes']
    sparse_prompt_embeds = save_sam_data['sparse_prompt_embeds']
    dense_prompt_embeds = save_sam_data['dense_prompt_embeds']
    image_embeddings = save_sam_data['q_image_embeds']
    
    cached = [[], []]
    if store_inp:
        data_saver_inp = DataSaverPreHook(stop_forward=False)
        handle_inp = module.register_forward_pre_hook(data_saver_inp)
        
    if store_oup:
        data_saver_oup = DataSaverHook(store_input=False, store_output=True, stop_forward=False)
        handle_oup = module.register_forward_hook(data_saver_oup)
        
    
    with torch.no_grad():
        for i in range(len(image_pes)):
            # print(i,len(cali_data))
            # _ = model.extract_feat(cali_data[i])
            mask_decoder.predict_calib_recon(image_embeddings=image_embeddings[i].to(device),
                                    image_pe = image_pes[i].to(device),
                                    sparse_prompt_embeddings = sparse_prompt_embeds[i].to(device), 
                                    dense_prompt_embeddings = dense_prompt_embeds[i].to(device))
            if store_inp:
                input_data = data_saver_inp.input_store
                
                if isinstance(input_data,tuple):
                    # print(len(input_data))
                    # print(len(input_data[0]))
                    if len(input_data) == 3:
                        cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                    elif len(input_data) == 4:  # per-stage
                        cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu(),input_data[3].cpu()))
                    elif len(input_data) == 2:
                        cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:   # len(input_data) == 1
                        if isinstance(input_data[0],tuple): # per-layer, qkv tuple
                            cached[0].append((input_data[0][0].cpu(),input_data[0][1].cpu(),input_data[0][2].cpu()))
                        elif len(input_data)==1:
                            cached[0].append(input_data[0].cpu())
                        else:
                            raise NotImplementedError('check data save')
                            
                else:
                    cached[0].append(input_data.cpu())
                    # print('tensor input')
                    
            if store_oup:
                if isinstance(data_saver_oup.output_store,tuple):
                    output_data = data_saver_oup.output_store
                    if len(output_data) == 3:    # per-stage
                        cached[1].append((output_data[0].detach().cpu(),output_data[1].detach().cpu()))
                    elif len(output_data) == 2:     # per-layer
                        cached[1].append(output_data[0].detach().cpu())
                    # print(len(data_saver.output_store))
                    else:
                        raise NotImplementedError('check data save')
                else:
                    cached[1].append(data_saver_oup.output_store.detach().cpu())
    if store_inp:
        handle_inp.remove()
        
    if store_oup:
        handle_oup.remove()
    torch.cuda.empty_cache()
    return cached

def save_inp_oup_data_blk(model, module, save_sam_data, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):

    device = next(module.parameters()).device
    mask_decoder = model.predictor.model.mask_decoder
    image_encoder = model.predictor.model.image_encoder
    
    image_data = save_sam_data['image_data']
    image_pes = save_sam_data['image_pes']
    sparse_prompt_embeds = save_sam_data['sparse_prompt_embeds']
    dense_prompt_embeds = save_sam_data['dense_prompt_embeds']
    # image_embeddings = save_sam_data['q_image_embeds']
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for i in range(len(image_data)):
            # print(i,len(cali_data))
            try:
                image_embed = image_encoder(image_data[i].to(device))
                mask_decoder.predict_calib_recon(image_embeddings=image_embed.to(device),
                                    image_pe = image_pes[i].to(device),
                                    sparse_prompt_embeddings = sparse_prompt_embeds[i].to(device), 
                                    dense_prompt_embeddings = dense_prompt_embeds[i].to(device))
                # _ = model.extract_feat(cali_data[i])
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0])
                else:
                    # print(type(data_saver.input_store))
                    # print(len(data_saver.input_store))
                    input_data = data_saver.input_store[0]
                    if isinstance(input_data,tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                        else:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    if isinstance(data_saver.output_store,tuple):
                        cached[1].append(data_saver.output_store[0].detach().cpu())
                        # print(len(data_saver.output_store))
                    else:
                        cached[1].append(data_saver.output_store.detach().cpu())
                    # cached[1].append(data_saver.output_store[0].detach().cpu())
    # if store_inp:
    #     cached[0] = torch.cat([x for x in cached[0]])
    # if store_oup:
    #     cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached

def save_inp_oup_data(model, module, cali_data, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):

    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            try:
                _ = model.extract_feat(cali_data[i])
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0])
                else:
                    # print(type(data_saver.input_store))
                    # print(len(data_saver.input_store))
                    input_data = data_saver.input_store[0]
                    if isinstance(input_data,tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                        else:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    if isinstance(data_saver.output_store,tuple):
                        cached[1].append(data_saver.output_store[0].detach().cpu())
                        # print(len(data_saver.output_store))
                    else:
                        cached[1].append(data_saver.output_store.detach().cpu())
                    # cached[1].append(data_saver.output_store[0].detach().cpu())
    # if store_inp:
    #     cached[0] = torch.cat([x for x in cached[0]])
    # if store_oup:
    #     cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached

def save_inp_oup_data_en(sam, module, image_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):

    device = next(sam.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for i in range(len(image_data)):
            # print(i,len(cali_data))
            try:
                _ = sam.image_encoder(image_data[i].to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0])
                else:
                    # print(type(data_saver.input_store))
                    # print(len(data_saver.input_store))
                    input_data = data_saver.input_store[0]
                    if isinstance(input_data,tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                        else:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    if isinstance(data_saver.output_store,tuple):
                        cached[1].append(data_saver.output_store[0].detach().cpu())
                        # print(len(data_saver.output_store))
                    else:
                        cached[1].append(data_saver.output_store.detach().cpu())
                    # cached[1].append(data_saver.output_store[0].detach().cpu())
    # if store_inp:
    #     cached[0] = torch.cat([x for x in cached[0]])
    # if store_oup:
    #     cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
        
        

class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 args,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 reg_weight=None,
                 reg_weight_lamb=0.1,
                 recon_loss='ori'
                 ):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p
        self.args = args
        self.recon_loss = recon_loss
        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.reg_weight=reg_weight
        self.reg_weight_lamb = reg_weight_lamb
        self.mse = torch.nn.MSELoss()
        
        # self.KLD_loss = torch.nn.KLDivLoss(reduction='batchmean')
    

    def __call__(self, pred, tgt, pred_attention=None, target_attention=None, T=1, W = None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        
        self.count += 1
        
        if pred_attention == None:
            if isinstance(pred, tuple):
                rec_loss = 0.01*lp_loss(pred[0], tgt[0], p=self.p)+lp_loss(pred[1], tgt[1])
            else:
                rec_loss = lp_loss(pred, tgt, p=self.p)
        else:
            rec_loss = 0


        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
            w_reg_loss = 0
        else:
            round_loss = 0
            w_reg_loss = 0
            layer_len = 0

        if pred_attention != None:
            if self.args.CAM_loss=='PAR':
                # rec_loss = 0
                # CAM_sim_loss = lp_loss(pred_attention, target_attention, p=2.0, sum_dim = -1)
                if self.recon_loss=='encoder':
                    # rec_loss = 0
                    rec_loss = lp_loss(pred, tgt, p=self.p)
                    CAM_sim_loss = lp_loss(pred_attention, target_attention, p=2.0, sum_dim = -1)
                    total_loss=CAM_sim_loss
                elif self.recon_loss=='decoder':
                    if isinstance(pred, tuple):
                        rec_loss = 0.01*lp_loss(pred[0], tgt[0], p=self.p)+lp_loss(pred[1], tgt[1])
                    else:
                        rec_loss = lp_loss(pred, tgt, p=self.p)
                    CAM_sim_loss = lp_loss(pred_attention, target_attention, p=2.0, sum_dim = -1)
                    total_loss = self.args.beta*rec_loss + self.args.alpha*CAM_sim_loss
                elif self.recon_loss=='ori':
                    rec_loss = lp_loss(pred, tgt, p=self.p)
                    CAM_sim_loss=0
                    total_loss=rec_loss
                else:
                    raise NotImplementedError("please check the CAM_loss")
            elif self.args.CAM_loss=='ori':
                CAM_sim_loss=0
                rec_loss = lp_loss(pred, tgt, p=self.p)
                total_loss=rec_loss
            else:
                raise NotImplementedError("please check the CAM_loss")
        else:
            CAM_sim_loss = 0
            # total_loss = rec_loss + round_loss
            total_loss = rec_loss

        if self.count % 100 == 0:
            logger.info('Total loss:\t{:.4f} (rec:{:.4f}, round:{:.4f}, rw:{:.4f}, CAM_sim:{:.10f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), float(w_reg_loss), float(CAM_sim_loss), b, self.count))
            # logger.info('AOL_loss_hard={:4f}'.format(CAM_sim_loss_h))
        # return CAM_sim_loss
        if self.args.use_wandb:
            try:
                wandb.log({'iters':self.count,'total_loss':total_loss, 'rec_loss':rec_loss, 'round_loss':round_loss, 'CAM_sim_loss':CAM_sim_loss})
            except:
                pass
            
        return total_loss


def lp_loss(pred, tgt, p=2.0, sum_dim = -1):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(sum_dim).mean()




def save_sam_inputs(model, cali_data, max_pt = 32):
    # model.cuda(1)
    inputs = []
    # prompt_tokens = []
    image_pes = []
    sparse_prompt_embeds = []
    dense_prompt_embeds = []
    
    image_encoder = model.predictor.model.image_encoder
    mask_decoder=model.predictor.model.mask_decoder
    prompt_encoder=model.predictor.model.prompt_encoder
    device = next(model.parameters()).device
    
    data_saver_pe = DataSaverHook(store_input=False, store_output=True, stop_forward=False)
    handle_pe = prompt_encoder.register_forward_hook(data_saver_pe)
    data_saver_im = DataSaverHook(store_input=True, store_output=False, stop_forward=False)
    handle_im = image_encoder.register_forward_hook(data_saver_im)
    
    # 提取标定集的prompt embedding
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            try:
                _ = model.extract_feat(cali_data[i])
            except StopForwardException:
                pass
            image_data = data_saver_im.input_store[0].detach().cpu()
            sparse_prompt_embed = data_saver_pe.output_store[0].detach().cpu()
            dense_prompt_embed = data_saver_pe.output_store[1].detach().cpu()
            image_pe = prompt_encoder.get_dense_pe().detach().cpu()
            
            
            B = sparse_prompt_embed.size(0)
    
            # 如果batch size小于等于目标大小，直接返回原tensor
            if B > max_pt:
                # 随机选择target_batch_size个索引
                indices = torch.randperm(B)[:max_pt]
                
                # 根据索引选择样本
                sparse_prompt_embed = sparse_prompt_embed[indices]
                dense_prompt_embed = dense_prompt_embed[indices]
            inputs.append(image_data)
            sparse_prompt_embeds.append(sparse_prompt_embed)
            dense_prompt_embeds.append(dense_prompt_embed)
            image_pes.append(image_pe)
            
    handle_pe.remove()
    handle_im.remove()
    model.cuda(0)
    torch.cuda.empty_cache()
    

    return inputs, image_pes, sparse_prompt_embeds, dense_prompt_embeds


def transform_image_token(tokens_in, neck):
    # 将每个stage输出的image token通过neck降维，支持attention block和mlp的输出激活
    if type(tokens_in) == list: # fp
        image_tokens = []
        for token in tokens_in:
            with torch.no_grad():
                if token.shape[1] == 14:
                    token = window_unpartition(token, 14, (70, 70), (64, 64))
                img_token = neck(token.permute(0, 3, 1, 2).cuda())

            image_tokens.append(img_token.detach().cpu())

        return image_tokens
        
    else:   # quant
        device = tokens_in.device
        token_in = tokens_in
        if token_in.shape[1] == 14:
            token_in = window_unpartition(token_in, 14, (70, 70), (64, 64))
        img_token = neck(token_in.permute(0, 3, 1, 2).cuda())

        return img_token.to(device)



    
def save_CIT_list(image_embeddings, image_pes, sparse_prompt_embeds, dense_prompt_embeds, mask_decoder):
    # 保存FP的cross image token作为监督，不计算梯度
    CA_block = mask_decoder.transformer
    dc_out_list = []
    
    data_saver_pe = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
    handle = CA_block.register_forward_hook(data_saver_pe)
    
    with torch.no_grad():
        for i in range(len(image_embeddings)):
            try:
                mask_decoder.predict_calib_recon(image_embeddings=image_embeddings[i].cuda(),
                                           image_pe = image_pes[i].cuda(),
                                           sparse_prompt_embeddings = sparse_prompt_embeds[i].cuda(), 
                                           dense_prompt_embeddings = dense_prompt_embeds[i].cuda(),
                                           calib_num=16)
            except StopForwardException:
                pass
            dc_out = data_saver_pe.output_store[1]
            # if dc_out.shape[0]>16:
            #     dc_out = dc_out[0:16,:,:]
            dc_out_list.append(dc_out.detach().cpu())
    
    handle.remove()
    torch.cuda.empty_cache()
    
    return dc_out_list

def get_CIT(image_embedding, image_pe, sparse_prompt_embed, dense_prompt_embed, mask_decoder,recon_gpu=0):
    # 获取cross image token，计算梯度
    CA_block = mask_decoder.transformer
    data_saver_pe = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
    
    handle = CA_block.register_forward_hook(data_saver_pe)
    try:
        mask_decoder.predict_calib_recon(image_embeddings=image_embedding.cuda(recon_gpu),
                                    image_pe = image_pe.cuda(recon_gpu),
                                    sparse_prompt_embeddings = sparse_prompt_embed.cuda(recon_gpu), 
                                    dense_prompt_embeddings = dense_prompt_embed.cuda(recon_gpu),
                                    calib_num=16)
    except StopForwardException:
        pass
    
    dc_out = data_saver_pe.output_store[1]

    handle.remove()
    torch.cuda.empty_cache()

    
    return dc_out

def reconstruction_IE(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, Aux_module, name):
    
    if name == 'neck':
        reconstruction_blk(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, name)
        return
    # else:
    #     return
    model.det_model.cpu()
    fp_model.det_model.cpu()
    
    image_data = save_sam_data['image_data']
    image_pes = save_sam_data['image_pes']
    sparse_prompt_embeds = save_sam_data['sparse_prompt_embeds']
    dense_prompt_embeds = save_sam_data['dense_prompt_embeds']
    # image_pes, sparse_prompt_embeds, dense_prompt_embeds = save_auxiliary_token(fp_model, cali_data)
    
    # get data first
    quant_inp, _ = save_inp_oup_data_en(model.predictor.model, module, image_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data_en(fp_model.predictor.model, fp_module, image_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    
    if args.CAM_loss=='PAR':
        image_tokens_fp = transform_image_token(fp_oup, fp_model.predictor.model.image_encoder.neck)
        # save interaction response target
        fp_dc_out = save_CIT_list(image_tokens_fp, image_pes, sparse_prompt_embeds, dense_prompt_embeds, fp_model.predictor.model.mask_decoder)
    
    

    from mmdet.utils import build_ddp,build_dp
    import os
    # from torch.utils.checkpoint import checkpoint
    # module.gradient_checkpointing_enable()
    recon_gpu = 1 if args.gpu2 else 0
    module_ddp = build_dp(module, 'cuda', device_ids=[recon_gpu])
    model.cuda(recon_gpu)
    
    # print(group_id_list)
    w_para, a_para = [], []
    torch.cuda.empty_cache()
    enable_quantization(module)
    
    for name_m, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (config.only4 and ('k_proj' in name_m or 'q_proj' in name_m))
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name_m:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
    
    # print(w_para, a_para)
    # exit()
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        # a_opt = torch.optim.Adam([{"params":a_para,"lr":config.scale_lr},{"params":module.gamma,"lr":config.scale_lr}])
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters_CAMS, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    # if len(gamma_para) != 0:
    #     gamma_opt = torch.optim.Adam(gamma_para, lr=config.gamma_lr)
    # else:
    #     gamma_opt = None
    
    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))


    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp,fp_oup,quant_inp
        torch.cuda.empty_cache()
        for name_m, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name_m:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(args, module=module, weight=config.weight, iters=config.iters_CAMS, b_range=config.b_range,
                            warm_up=config.warm_up, recon_loss='encoder')

    sz = len(cali_data)
    for i in range(config.iters_CAMS):
        idx = torch.randint(0, sz, (1, ))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]
    
            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda(recon_gpu)
                cur_fp_inp = cur_fp_inp.cuda(recon_gpu)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp.cuda(recon_gpu))
            elif len(cur_quant_inp) == 2:
                
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0,cur_inp1)
            else:
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu), cur_quant_inp[2].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu), cur_fp_inp[2].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2], cur_fp_inp[2])
                cur_inp = (cur_inp0,cur_inp1,cur_inp2)
        else:
            cur_inp = quant_inp[idx].cuda(recon_gpu)
        
        cur_fp_oup = fp_oup[idx].cuda(recon_gpu)
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        try:
            cur_quant_oup, _ = module_ddp(cur_inp.cuda(recon_gpu))
        except:
            cur_quant_oup = module_ddp(cur_inp.cuda(recon_gpu))
        
        if args.CAM_loss=='PAR':
            if name == 'neck':
                cur_image_tokens_q = cur_quant_oup
            else:
                cur_image_tokens_q = transform_image_token(cur_quant_oup, fp_model.predictor.model.image_encoder.neck)
            
            # get quantization interaction response
        
            cur_dc_out = get_CIT(
                image_embedding=cur_image_tokens_q, 
                image_pe=image_pes[idx], 
                sparse_prompt_embed=sparse_prompt_embeds[idx],
                dense_prompt_embed=dense_prompt_embeds[idx],
                mask_decoder=fp_model.predictor.model.mask_decoder.cuda(recon_gpu),
                recon_gpu=recon_gpu
                )
            
            err = loss_func(cur_quant_oup, cur_fp_oup, cur_dc_out.cuda(recon_gpu), fp_dc_out[idx].cuda(recon_gpu))
            del cur_inp, cur_quant_oup
            torch.cuda.empty_cache()
            err.backward() # del cur_inp cur_quant_oup
        # rank, world_size = get_dist_info()
        elif args.CAM_loss == 'ori':
            err = loss_func(cur_quant_oup, cur_fp_oup)
            torch.cuda.empty_cache()
            err.backward() # del cur_inp cur_quant_oup
        else:
            raise NotImplementedError('check args.CAM_loss')
        
        
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()
        
        torch.cuda.empty_cache()
        
    w_para, a_para = [], []
    del w_opt, a_opt, a_scheduler, cur_fp_oup, err
    torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
        
    del fp_inp,fp_oup,quant_inp
    module.cuda(0)
    model.cuda(0)
    fp_model.cuda(0)
    if args.gpu2:
        clear_gpu_memory(gpu_id=1)
    
    torch.cuda.empty_cache()
        
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0
    


def reconstruction_IE_PB(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, Aux_module, name):
    
    if name == 'neck':
        reconstruction_blk(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, name)
        return
    # else:
    #     return
    model.det_model.cpu()
    fp_model.det_model.cpu()
    
    image_data = save_sam_data['image_data']
    image_pes = save_sam_data['image_pes']
    sparse_prompt_embeds = save_sam_data['sparse_prompt_embeds']
    dense_prompt_embeds = save_sam_data['dense_prompt_embeds']
    # image_pes, sparse_prompt_embeds, dense_prompt_embeds = save_auxiliary_token(fp_model, cali_data)
    
    # get data first
    quant_inp, _ = save_inp_oup_data_en(model.predictor.model, module, image_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data_en(fp_model.predictor.model, fp_module, image_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    
    image_tokens_fp = transform_image_token(fp_oup, fp_model.predictor.model.image_encoder.neck)
        
    fp_dc_out = save_CIT_list(image_tokens_fp, image_pes, sparse_prompt_embeds, dense_prompt_embeds, fp_model.predictor.model.mask_decoder)
    
    

    from mmdet.utils import build_ddp,build_dp
    import os
    # from torch.utils.checkpoint import checkpoint
    # module.gradient_checkpointing_enable()
    recon_gpu = 1 if args.gpu2 else 0
    module_ddp = build_dp(module, 'cuda', device_ids=[recon_gpu])
    model.cuda(recon_gpu)
    # module_ddp = build_dp(module, 'cuda', device_ids=[0,1])
    # module_ddp = module.cuda(1)
    
    # print(group_id_list)
    w_para, a_para = [], []
    torch.cuda.empty_cache()
    enable_quantization(module)
    
    for name_m, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (config.only4 and ('k_proj' in name_m or 'q_proj' in name_m))
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name_m:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
    
    # print(w_para, a_para)
    # exit()
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        # a_opt = torch.optim.Adam([{"params":a_para,"lr":config.scale_lr},{"params":module.gamma,"lr":config.scale_lr}])
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters_CAMS, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    # if len(gamma_para) != 0:
    #     gamma_opt = torch.optim.Adam(gamma_para, lr=config.gamma_lr)
    # else:
    #     gamma_opt = None
    
    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))


    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp,fp_oup,quant_inp
        torch.cuda.empty_cache()
        for name_m, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name_m:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(args, module=module, weight=config.weight, iters=config.iters_CAMS, b_range=config.b_range,
                            warm_up=config.warm_up, recon_loss='encoder')

   
    sz = len(cali_data)
    for i in range(config.iters_CAMS):
        idx = torch.randint(0, sz, (1, ))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]
    
            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda(recon_gpu)
                cur_fp_inp = cur_fp_inp.cuda(recon_gpu)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp.cuda(recon_gpu))
            elif len(cur_quant_inp) == 2:
                
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0,cur_inp1)
            else:
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu), cur_quant_inp[2].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu), cur_fp_inp[2].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2], cur_fp_inp[2])
                cur_inp = (cur_inp0,cur_inp1,cur_inp2)
        else:
            cur_inp = quant_inp[idx].cuda(recon_gpu)
        
        cur_fp_oup = fp_oup[idx].cuda(recon_gpu)
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        try:
            cur_quant_oup, _ = module_ddp(cur_inp.cuda(recon_gpu))
        except:
            cur_quant_oup = module_ddp(cur_inp.cuda(recon_gpu))
            
        if name == 'neck':
            cur_image_tokens_q = cur_quant_oup
        else:
            cur_image_tokens_q = transform_image_token(cur_quant_oup, fp_model.predictor.model.image_encoder.neck)
        
    
        cur_dc_out = get_CIT(
            image_embedding=cur_image_tokens_q, 
            image_pe=image_pes[idx], 
            sparse_prompt_embed=sparse_prompt_embeds[idx],
            dense_prompt_embed=dense_prompt_embeds[idx],
            mask_decoder=fp_model.predictor.model.mask_decoder.cuda(recon_gpu),
            recon_gpu=recon_gpu
            )
        
        err = loss_func(cur_quant_oup, cur_fp_oup, cur_dc_out.cuda(recon_gpu), fp_dc_out[idx].cuda(recon_gpu))
        del cur_inp, cur_quant_oup
        rank, world_size = get_dist_info()
        
        torch.cuda.empty_cache()
        err.backward() # del cur_inp cur_quant_oup
        
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()
        
        torch.cuda.empty_cache()
        
    w_para, a_para = [], []
    del w_opt, a_opt, a_scheduler, cur_fp_oup, err
    torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
        
    del fp_inp,fp_oup,quant_inp
    module.cuda(0)
    model.cuda(0)
    fp_model.cuda(0)
    if args.gpu2:
        clear_gpu_memory(gpu_id=1)
    
    torch.cuda.empty_cache()
        
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0



def save_dc_Qout_list(image_embeddings, image_pes, sparse_prompt_embeds, dense_prompt_embeds, mask_decoder):
    FA_block = mask_decoder.transformer
    Qout_list = []
    data_saver_pe = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
    handle = FA_block.register_forward_hook(data_saver_pe)
    with torch.no_grad():
        for i in range(len(image_embeddings)):
            try:
               mask_decoder.predict_calib_recon(image_embeddings=image_embeddings[i].cuda(),
                                    image_pe = image_pes[i].cuda(),
                                    sparse_prompt_embeddings = sparse_prompt_embeds[i].cuda(), 
                                    dense_prompt_embeddings = dense_prompt_embeds[i].cuda())
            except StopForwardException:
                pass
            # T = 1
            Qout = data_saver_pe.output_store[0]
                
            Qout_list.append(Qout.detach().cpu())
            
    handle.remove()
    torch.cuda.empty_cache()
    return Qout_list




def get_q_image_embedding(args, q_model, image_data):
    
    FA_block = q_model.predictor.model.image_encoder
    device = next(FA_block.parameters()).device
    data_saver_pe = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
    handle = FA_block.register_forward_hook(data_saver_pe)
    q_image_embed_list = []
    with torch.no_grad():
        for i in range(len(image_data)):
            try:
                _ = FA_block(image_data[i].to(device))
            except StopForwardException:
                pass
            q_image_embed = data_saver_pe.output_store
            q_image_embed_list.append(q_image_embed)
            # print(q_image_embed.shape)
    handle.remove()
    torch.cuda.empty_cache()
    return q_image_embed_list



def get_dc_Qout(image_embedding, image_pe, sparse_prompt_embed, dense_prompt_embed, mask_decoder,recon_gpu=0):
    
    # FA_block = mask_decoder.transformer.layers[0].cross_attn_token_to_image
    FA_block = mask_decoder.transformer
    data_saver_pe = DataSaverHook(store_input=False, store_output=True, stop_forward=False)
    
    handle = FA_block.register_forward_hook(data_saver_pe)
    try:
        mask_decoder.predict_calib_recon(image_embeddings=image_embedding.cuda(recon_gpu),
                                    image_pe = image_pe.cuda(recon_gpu),
                                    sparse_prompt_embeddings = sparse_prompt_embed.cuda(recon_gpu), 
                                    dense_prompt_embeddings = dense_prompt_embed.cuda(recon_gpu))
    except StopForwardException:
        pass
    
    # T = 1
    
    Qout = data_saver_pe.output_store[0]
    
    handle.remove()
    torch.cuda.empty_cache()
    # print(CAM.shape)
    # exit()
    # print('CAM:',CAM.requires_grad)
    
    return Qout


def reconstruction_MD(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, Aux_module, name):
    
    if name == 'final_attn_token_to_image':
        # set more reconstruction iteration for final layer, because it has significant higher loss
        config.iters = 10000
        reconstruction_blk(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, name)
        return
    # config.iters_CAMS = 3000
    model.det_model.cpu()
    fp_model.det_model.cpu()
    
    image_data = save_sam_data['image_data']
    image_pes = save_sam_data['image_pes']
    sparse_prompt_embeds = save_sam_data['sparse_prompt_embeds']
    dense_prompt_embeds = save_sam_data['dense_prompt_embeds']
    q_image_embeds = get_q_image_embedding(args, model, image_data)
    save_sam_data['q_image_embeds'] = q_image_embeds
    # image_pes, sparse_prompt_embeds, dense_prompt_embeds = save_auxiliary_token(model, cali_data)
    # q_image_embeds, image_pes, sparse_prompt_embeds, dense_prompt_embeds = get_q_imembed_aux_token(args, model, cali_data)
    
    device = next(module.parameters()).device
    # fp_QKcos_list = save_dc_Qout_list(args, fp_model, cali_data)
    if args.CAM_loss=='PAR':
        fp_QKcos_list = save_dc_Qout_list(q_image_embeds, image_pes, sparse_prompt_embeds, dense_prompt_embeds, fp_model.predictor.model.mask_decoder)
    # cross_token_list, image_pe_list, token_pe_list = save_cross_token(fp_model, cali_data)
    
    # get data first
    quant_inp, _ = save_inp_oup_data_MD(model, module, save_sam_data, store_inp=True, store_oup=False)
    fp_inp, fp_oup = save_inp_oup_data_MD(fp_model, fp_module, save_sam_data, store_inp=True, store_oup=True)

    w_para, a_para = [], []
    
    # # for the bimodal block, add the gamma parameter
    # gamma_para = []
    # if hasattr(module,'gamma') and config.gamma_tune:
    #     gamma_para.append(module.gamma)
    torch.cuda.empty_cache()
    for name_m, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (config.only4 and ('k_proj' in name_m or 'q_proj' in name_m))
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name_m:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
    # print(w_para, a_para)
    # exit()
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        # a_opt = torch.optim.Adam([{"params":a_para,"lr":config.scale_lr},{"params":module.gamma,"lr":config.scale_lr}])
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters_CAMS, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    # if len(gamma_para) != 0:
    #     gamma_opt = torch.optim.Adam(gamma_para, lr=config.gamma_lr)
    # else:
    #     gamma_opt = None
    
    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))


    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp,fp_oup,quant_inp
        torch.cuda.empty_cache()
        for name_m, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name_m:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(args, module=module, weight=config.weight, iters=config.iters_CAMS, b_range=config.b_range,
                             warm_up=config.warm_up, recon_loss='decoder')

    from mmdet.utils import build_ddp,build_dp
    import os
    # module_ddp = build_dp(module, 'cuda', device_ids=[0,1])
    recon_gpu = 1 if args.gpu2 else 0
    module_ddp = build_dp(module, 'cuda', device_ids=[recon_gpu])

    sz = len(cali_data)
    for i in range(config.iters_CAMS):
        idx = torch.randint(0, sz, (1, ))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]
    
            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda(recon_gpu)
                cur_fp_inp = cur_fp_inp.cuda(recon_gpu)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp.cuda(recon_gpu))
            elif len(cur_quant_inp) == 2:
                
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0,cur_inp1)
            elif len(cur_quant_inp) == 3:
                # print('len(cur_quant_inp[0]) == 3')
                
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu), cur_quant_inp[2].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu), cur_fp_inp[2].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2], cur_fp_inp[2])
                cur_inp = (cur_inp0,cur_inp1,cur_inp2)
            else: # len(cur_quant_inp[0]) == 4
                # print('len(cur_quant_inp[0]) == 4')
                # print(len(cur_quant_inp))
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0.cuda(recon_gpu), cur_inp1.cuda(recon_gpu),cur_quant_inp[2].cuda(recon_gpu),cur_quant_inp[3].cuda(recon_gpu))
        else:
            cur_inp = quant_inp[idx]
            
        if args.recon_unit=='per_block':
            cur_fp_oup=fp_oup[idx].cuda(recon_gpu)
        else:
            cur_fp_oup = (fp_oup[idx][0].cuda(recon_gpu), fp_oup[idx][1].cuda(recon_gpu))
            
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        
        if args.recon_unit=='per_block':
            try:
                cur_quant_outputs,_ = module_ddp(cur_inp)
            except:
                cur_quant_outputs = module_ddp(cur_inp)
            cur_quant_oup = cur_quant_outputs
        else:
            cur_quant_outputs = module_ddp(*cur_inp)
            cur_quant_oup = (cur_quant_outputs[0], cur_quant_outputs[1])
        
        # if args.recon_unit=='per_block':
        #     cur_quant_oup = cur_quant_outputs
        # else:
        #     cur_quant_oup = (cur_quant_outputs[0], cur_quant_outputs[1])
        
        if args.CAM_loss=='PAR':
            cur_QK_cos = get_dc_Qout(
                image_embedding=q_image_embeds[idx], 
                image_pe=image_pes[idx], 
                sparse_prompt_embed=sparse_prompt_embeds[idx],
                dense_prompt_embed=dense_prompt_embeds[idx],
                mask_decoder=model.predictor.model.mask_decoder.cuda(recon_gpu),
                recon_gpu=recon_gpu
                )
            
            err = loss_func(cur_quant_oup, cur_fp_oup, cur_QK_cos, fp_QKcos_list[idx].cuda(recon_gpu))
            cur_inp = None
            cur_quant_oup = None
            cur_QK_cos = None
            torch.cuda.empty_cache()
            err.backward() # del cur_inp cur_quant_oup
            del cur_inp, cur_quant_oup, cur_QK_cos, err

        elif args.CAM_loss == 'ori':
            err = loss_func(cur_quant_oup, cur_fp_oup)
            err.backward() # del cur_inp cur_quant_oup
            del cur_inp, cur_quant_oup, err
        else:
            raise NotImplementedError('check args.CAM_loss')  
        
        
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()
        
        torch.cuda.empty_cache()
 
    
    del fp_inp,fp_oup,quant_inp,cur_fp_oup
    torch.cuda.empty_cache()
    module.cuda(0)
    model.predictor.model.mask_decoder.cuda(0)
    
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0


def reconstruction_blk(args, model, fp_model, module, fp_module, cali_data, save_sam_data, config, name):
    device = next(module.parameters()).device
    if name == 'final_attn_token_to_image':
        # set more reconstruction iteration for final layer, because it has significant higher loss
        config.iters = 10000
    # enable_quantization(model)
    # get data first
    model.det_model.cuda()
    fp_model.det_model.cuda()
    quant_inp, _ = save_inp_oup_data_blk(model, module, save_sam_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data_blk(fp_model, fp_module, save_sam_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    w_para, a_para = [], []
    
    # # for the bimodal block, add the gamma parameter
    # gamma_para = []
    # if hasattr(module,'gamma') and config.gamma_tune:
    #     gamma_para.append(module.gamma)

    for name, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (config.only4 and ('k_proj' in name or 'q_proj' in name))
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
    # print(w_para, a_para)
    # exit()
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        # a_opt = torch.optim.Adam([{"params":a_para,"lr":config.scale_lr},{"params":module.gamma,"lr":config.scale_lr}])
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    # if len(gamma_para) != 0:
    #     gamma_opt = torch.optim.Adam(gamma_para, lr=config.gamma_lr)
    # else:
    #     gamma_opt = None
    
    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))


    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp,fp_oup,quant_inp
        torch.cuda.empty_cache()
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(args, module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up)

    from mmdet.utils import build_ddp,build_dp
    import os
    recon_gpu = 1 if args.gpu2 else 0
    module_ddp = build_dp(module, 'cuda', device_ids=[recon_gpu])

        
    sz = len(cali_data)
    for i in range(config.iters):
        idx = torch.randint(0, sz, (1, ))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]
    
            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda(recon_gpu)
                cur_fp_inp = cur_fp_inp.cuda(recon_gpu)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp.cuda(recon_gpu))
            elif len(cur_quant_inp) == 2:
                
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp = (cur_inp0,cur_inp1)
            else:
                cur_quant_inp = (cur_quant_inp[0].cuda(recon_gpu), cur_quant_inp[1].cuda(recon_gpu), cur_quant_inp[2].cuda(recon_gpu))
                cur_fp_inp = (cur_fp_inp[0].cuda(recon_gpu), cur_fp_inp[1].cuda(recon_gpu), cur_fp_inp[2].cuda(recon_gpu))
                
                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0], cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1], cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2], cur_fp_inp[2])
                cur_inp = (cur_inp0,cur_inp1,cur_inp2)
        else:
            cur_inp = quant_inp[idx]
            
        cur_fp_oup = fp_oup[idx].cuda(recon_gpu)
            
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        try:
            cur_quant_oup, _ = module_ddp(cur_inp)
        except:
            cur_quant_oup = module_ddp(cur_inp)
            
        err = loss_func(cur_quant_oup, cur_fp_oup.cuda(recon_gpu))
        cur_inp = None
        cur_quant_oup = None
        torch.cuda.empty_cache()
        err.backward() # del cur_inp cur_quant_oup
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()
            
    
    del fp_inp,fp_oup,quant_inp,cur_fp_oup
    module.cuda(0)
    model.cuda(0)
    torch.cuda.empty_cache()
    
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def apply_QKst_tmp(model, scales):
    with torch.no_grad():
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
    
        
    model.q_proj.module.use_temporary_parameter = True
    model.k_proj.module.use_temporary_parameter = True
    model.q_proj.module.temp_weight = model.q_proj.module.weight/scales.view(-1,1)
    model.q_proj.module.temp_bias = model.q_proj.module.bias/scales.view(-1)
    model.k_proj.module.temp_weight = model.k_proj.module.weight*scales.view(-1,1)
    model.k_proj.module.temp_bias = model.k_proj.module.bias*scales.view(-1)


def sync_gradients(model, world_size):
    """同步模型梯度"""
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            # 首先将梯度除以进程数
            param.grad.data /= world_size
            # 在所有进程间同步梯度
            dist.all_reduce(param.grad.data)

def clear_gpu_memory(gpu_id=None):
    """
    清理指定GPU的显存缓存
    Args:
        gpu_id: 要清理的GPU ID。如果为None，则清理当前设备
    """
    if gpu_id is not None:
        original_device = torch.cuda.current_device()
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        gc.collect()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
            except Exception:
                pass
        torch.cuda.empty_cache()
        torch.cuda.set_device(original_device)
    else:
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
            except Exception:
                pass