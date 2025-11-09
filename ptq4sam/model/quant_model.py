import math
import pandas as pd
import torch
import torch.nn as nn
import functools
import logging
import numpy as np
from torch import Tensor
from typing import Tuple
from scipy import stats
from scipy.stats import gaussian_kde
from ptq4sam.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer, QuantizedTransformerLayer, QuantizedTransformerStage   # noqa: F401
from ptq4sam.quantization.fake_quant import QuantizeBase   # noqa: F401
from projects.instance_segment_anything.models.segment_anything.modeling.transformer import Attention, TwoWayAttentionBlock
from projects.instance_segment_anything.models.segment_anything.modeling.common import MLPBlock
from ptq4sam.quantization.quantized_module import PreQuantizedLayer,QuantizedMatMul

from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import Attention as EncoderAttention
from projects.instance_segment_anything.models.segment_anything.modeling.transformer import Attention as DecoderAttention
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import Block, ImageEncoderViT
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import window_partition, window_unpartition, add_decomposed_rel_pos
# from .fake_quant import QuantizeBase
logger = logging.getLogger('ptq4sam')
def update_specialized_quantizer_config(base_config, quantizer_name):
    import copy
    specialized_config = copy.deepcopy(base_config)

    update_keys = {
        'softmax':{'quantizer':'AdaptiveGranularityQuantize',
                   'observer':'LogAvgMSEFastObserver'},
        'bimodal':{'quantizer':'LSQSignFakeQuantize',
                   'observer':'SignAvgMSEFastObserver'}
    }[quantizer_name]
    specialized_config.update(update_keys)
    return specialized_config

class AttentionOverlapLoss(nn.Module):
    def __init__(self, threshold: float = 0.5):
        """
        注意力图重叠度损失函数
        
        Args:
            threshold: 阈值，用于确定高注意力区域
        """
        super().__init__()
        self.threshold = threshold
        
    def forward(self, attn1: torch.Tensor, attn2: torch.Tensor) -> torch.Tensor:
        """
        计算两个注意力图之间的损失
        
        Args:
            attn1: 第一个注意力图 [batch_size, height, width] 或 [batch_size, 1, height, width]
            attn2: 第二个注意力图 [batch_size, height, width] 或 [batch_size, 1, height, width]
            
        Returns:
            loss: 损失值
        """
        if attn1.dim() == 3:
            attn1 = attn1.unsqueeze(1)
        if attn2.dim() == 3:
            attn2 = attn2.unsqueeze(1)
            
        # 获取每个batch中的最大值
        # max_vals1 = attn1.amax(dim=(2,3), keepdim=True)
        # max_vals2 = attn2.amax(dim=(2,3), keepdim=True)
        max_vals1 = attn1.amax(dim=(3), keepdim=True)
        max_vals2 = attn2.amax(dim=(3), keepdim=True)
        
        mask1 = (attn1 > self.threshold * max_vals1).float()
        mask2 = (attn2 > self.threshold * max_vals2).float()
        
        intersection = torch.min(mask1, mask2)
        union = torch.max(mask1, mask2)
        
        intersection_sum = intersection.sum(dim=(2,3))
        union_sum = union.sum(dim=(2,3))
        
        eps = 1e-6
        iou = intersection_sum / (union_sum + eps)
        
        # 计算IoU损失（1 - IoU）
        iou_loss = 1 - iou
        

        return iou_loss.mean()
    

class QuantEncoderAttentionBlock(QuantizedBlock):
    def __init__(self, org_module: EncoderAttention, w_qconfig, a_qconfig, qoutput=False, qinput=True):
        super().__init__()
        self.qinput = qinput
        self.qoutput = qoutput
        self.num_heads = org_module.num_heads
        self.scale = org_module.scale

        self.qkv = PreQuantizedLayer(org_module.qkv, None, w_qconfig, a_qconfig)
        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig)
        self.use_rel_pos = org_module.use_rel_pos

        if self.use_rel_pos:
            self.rel_pos_h = org_module.rel_pos_h
            self.rel_pos_w = org_module.rel_pos_w
        
        self.softmax_post_act_fake_quantize = Quantizer(None, a_qconfig)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
        
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        
        # print(q.shape, k.shape, v.shape, self.num_heads);exit()
        q = self.q_post_act_fake_quantize(q)
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = self.softmax_post_act_fake_quantize(attn.softmax(dim=-1))
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class QuantNeck(QuantizedBlock):
    def __init__(self, org_module: nn.Sequential, w_qconfig, a_qconfig, qoutput=True, qinput=False):
        super().__init__()
        org_module[0] = PreQuantizedLayer(org_module[0], None, w_qconfig, a_qconfig)
        org_module[2] = PreQuantizedLayer(org_module[2], None, w_qconfig, a_qconfig)
        self.model = org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QuantMLPBlock(QuantizedBlock):
    def __init__(self, org_module: MLPBlock, w_qconfig, a_qconfig, qinput=True):
        super().__init__()
        self.lin1 = PreQuantizedLayer(org_module.lin1, None, w_qconfig, a_qconfig)
        self.lin2 = PreQuantizedLayer(org_module.lin2, None, w_qconfig, a_qconfig)
        self.act = org_module.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class QunatEncoderBlock(nn.Module):
    def __init__(self, org_module: Block, w_qconfig, a_qconfig, qoutput=True ) -> None:
        super().__init__()
        self.norm1 = org_module.norm1
        self.attn = QuantEncoderAttentionBlock(org_module.attn, w_qconfig, a_qconfig)
        self.norm2 = org_module.norm2
        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig)

        self.window_size = org_module.window_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class QuantImageEncoderViT(nn.Module):
    
    def __init__(self, org_module: ImageEncoderViT, w_qconfig, a_qconfig, qoutput=True ) -> None:
        super().__init__()
        self.img_size = org_module.img_size
        self.patch_embed = org_module.patch_embed
        # do not quantize the first block/layer pos_embed
        self.pos_embed = org_module.pos_embed
        
        self.blocks = nn.ModuleList()
        for i in range(len(org_module.blocks)):
            self.blocks.append(QunatEncoderBlock(org_module.blocks[i], w_qconfig, a_qconfig))

        self.neck = QuantNeck(org_module.neck, w_qconfig, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
    

class QuantDecoderOurTwoWayAttentionBlock(QuantizedTransformerLayer):
# class QuantDecoderOurTwoWayAttentionBlock(nn.Module):
    
    def __init__(self, org_module: TwoWayAttentionBlock, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.self_attn = QuantDecoderOurAttentionBlock(org_module.self_attn, w_qconfig, a_qconfig, ptq4sam_config, qinput=True)
        self.norm1 = org_module.norm1

        self.cross_attn_token_to_image = QuantDecoderOurAttentionBlock(
            org_module.cross_attn_token_to_image, w_qconfig, a_qconfig, ptq4sam_config, qinput=True
        )
        self.norm2 = org_module.norm2

        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig, qinput=True)
        self.norm3 = org_module.norm3

        self.norm4 = org_module.norm4
        self.cross_attn_image_to_token = QuantDecoderOurAttentionBlock(
            org_module.cross_attn_image_to_token, w_qconfig, a_qconfig, ptq4sam_config, qinput=True
        )

        self.skip_first_layer_pe = org_module.skip_first_layer_pe
    
    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries, ATM_feature = self.self_attn((queries, queries, queries))
        else:
            q = queries + query_pe
            attn_out, ATM_feature = self.self_attn((q, q, queries))
            queries = queries + attn_out
        # feature_dic['SA'].append(ATM_feature[0])
        queries = self.norm1(queries)
        
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        
        # attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        attn_out, ATM_feature = self.cross_attn_token_to_image((q, k, keys))
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, ATM_feature = self.cross_attn_image_to_token((k, q, queries))
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys, None
    


class QuantDecoderOurAttentionBlock(QuantizedBlock):
    def __init__(self, org_module: DecoderAttention, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True, qinput=False, AttnRank=False):
        super().__init__()
        self.qoutput = qoutput
        self.embedding_dim = org_module.embedding_dim
        self.internal_dim = org_module.internal_dim
        self.num_heads = org_module.num_heads
        
        self.q_proj = PreQuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig)
        self.k_proj = PreQuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig)
        self.v_proj = PreQuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig)
        self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)
        # self.out_proj = QuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig, qoutput=False)

        self.AttnRank = AttnRank
        self.AttnRank_calib = True
        self.get_ori_ATM = False
        self.ori_ATM = None
        self.cnt = 0
        self.clip_l = torch.tensor(0.0).cuda()
        self.clip_r = torch.tensor(0.0).cuda()
        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax')
        else:
            softmax_a_config = a_qconfig
        if ptq4sam_config.BIG:
            sign_a_config = update_specialized_quantizer_config(a_qconfig, 'bimodal')
        else:
            sign_a_config = a_qconfig
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, sign_a_config)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

        if ptq4sam_config.BIG:
            self.k_post_act_fake_quantize.global_num = ptq4sam_config.global_num
            self.k_post_act_fake_quantize.peak_distance = ptq4sam_config.peak_distance
            self.k_post_act_fake_quantize.peak_height = ptq4sam_config.peak_height
            
        self.threshold=ptq4sam_config.threshold

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C


    def get_ATM(self, qkv):
        q,k,_ = qkv[0],qkv[1],qkv[2]
        q = self.q_post_act_fake_quantize(self.q_proj(q))
        k = self.k_post_act_fake_quantize(self.k_proj(k))
        # v = self.v_post_act_fake_quantize(self.v_proj(v))
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        # v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        
        attn = q @ k.permute(0, 1, 3, 2)
        
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        self.softmax_post_act_fake_quantize.disable_observer()
        attn = self.softmax_post_act_fake_quantize(attn)
        self.softmax_post_act_fake_quantize.enable_observer()

        # return A
        return attn
        

    def search_range_AOL(self,qkv,module,name):
        act_list = []
        AOL_loss = AttentionOverlapLoss(threshold=self.threshold)
        
        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            # 这里插入计算函数
            act_list.append(x)
        
        def calib_sz_across_pct(act, pct,calib=False):
            # print(act.shape)
            act_clone = act.clone().detach()
            try:
                clip_r = torch.quantile(act_clone.reshape(-1), pct)
                clip_l = torch.quantile(act_clone.reshape(-1), (1.0-pct))
            except:
                clip_r = torch.tensor(np.percentile(
                        act_clone.reshape(-1).cpu(), pct * 100),
                        device=act_clone.device,
                        dtype=torch.float32)
                clip_l = torch.tensor(np.percentile(
                        act_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=act_clone.device,
                        dtype=torch.float32)
            
            if calib:
                # if (self.cnt+1)%2 == 0:
                self.clip_l = self.clip_l * self.cnt + clip_l
                self.clip_r = self.clip_r * self.cnt + clip_r
                self.cnt += 1
                self.clip_l /= (self.cnt)
                self.clip_r /= (self.cnt)
                module.update_sz(self.clip_l, self.clip_r)
                
            else:
                module.update_sz(clip_l, clip_r)
        
        # hook input tensor in act_list
        hook_act = module.register_forward_hook(functools.partial(stat_input_hook, name=name))
        self.get_ATM(qkv)
        act_ori = act_list.pop()  # original input to be quantized   
        hook_act.remove()
        act_min, act_max = torch._aminmax(act_ori)
        best_score = torch.zeros_like(act_min)+(1e+10)
        
        best_pct = 1
        # percentage list to search
        pct_list = [0.85, 0.87, 0.88, 0.89,  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.960, 0.965, 0.970, 0.975, 0.980, 0.983, 0.985, 0.987, 0.989, 0.99, 0.993, 0.995, 0.997, 0.999, 0.9993, 0.9995, 0.9997, 0.9999, 0.99999]
        # search quantization parameter
        for pct in pct_list:
            calib_sz_across_pct(act_ori, pct)
            # print(module.scale)
            module.enable_fake_quant()
            q_ATM = self.get_ATM(qkv)
            _, _, h, w = q_ATM.shape
            score = AOL_loss(q_ATM, self.ori_ATM)
            module.disable_fake_quant()
            # print(score)
            if score < best_score:
                best_score = score
                best_pct = pct
            # print(pct, score)
                
        logger.info('best_pct:{}, best_score:{}'.format(best_pct, best_score.data.item()))
        calib_sz_across_pct(act_ori, best_pct, calib=False)
        del act_list
        del act_ori
        del q_ATM
        torch.cuda.empty_cache()
    
    
    def set_ori_ATM(self,qkv):
        self.ori_ATM = self.get_ATM(qkv)
            
    
    # def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    def forward(self, qkv: tuple) -> Tensor:
        
        logger = logging.getLogger('ptq4sam')    
        if self.AttnRank: 
                
            QKV_in_module = ['q_proj.layer_pre_act_fake_quantize',
                             'k_proj.layer_pre_act_fake_quantize']
            QKV_out_module = ['q_post_act_fake_quantize',
                              'k_post_act_fake_quantize']
            # Disable the observers for these KV activations, and the quantization parameters will be searched separately later.
            for name, submodule in self.named_modules():
                if isinstance(submodule, QuantizeBase):
                    if (name in QKV_in_module) or (name in QKV_out_module):
                    # if (name in QKV_in_module):
                        submodule.disable_observer()
            # Search quantization parameters based on Attention-IoU.
            if self.AttnRank_calib:
                logger.info('------------------')
                self.margin = 0
                logger.info('-----AOL calib-----')
                
                for name, submodule in self.named_modules():
                    if isinstance(submodule, QuantizeBase):
                        if name in QKV_in_module:
                            logger.info(name)
                            submodule.disable_observer()
                            self.search_range_AOL(qkv, submodule, name)
                            submodule.enable_fake_quant()
                            
                for name, submodule in self.named_modules():
                    if isinstance(submodule, QuantizeBase):
                        if name in QKV_out_module:
                            logger.info(name)
                            submodule.disable_observer()
                            self.search_range_AOL(qkv, submodule, name)
                            submodule.enable_fake_quant()
                            
                
                torch.cuda.empty_cache()
                # del self.ori_ATM
                # Calibration only once.
                self.AttnRank_calib = False
            
            q,k,v = qkv[0],qkv[1],qkv[2]                      
            # Input projections
            q_ = self.q_post_act_fake_quantize(self.q_proj(q))
            k_ = self.k_post_act_fake_quantize(self.k_proj(k))
            v_ = self.v_post_act_fake_quantize(self.v_proj(v))
            # print(q_.shape, k_.shape, v_.shape)
            # exit()

            # Separate into heads
            q = self._separate_heads(q_, self.num_heads)
            k = self._separate_heads(k_, self.num_heads)
            v = self._separate_heads(v_, self.num_heads)
            
            # Attention
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn / math.sqrt(c_per_head)
            QK = attn.clone()
            
            attn = torch.softmax(attn, dim=-1)
            
            attn = self.softmax_post_act_fake_quantize(attn,value=v)

            # Get output
            out = attn @ v
            
            out = self._recombine_heads(out)
            # print(out.shape)
            out = self.out_proj(out)
            
            

            return out, (QK, q_.clone(), k_.clone(), v_.clone())
                   
        else:
            if self.get_ori_ATM:
                logger.info('set ori ATM')
                self.set_ori_ATM(qkv)
                self.get_ori_ATM = False
            # self.ori_ARR = self.get_ARR(qkv)
            
            q,k,v = qkv[0],qkv[1],qkv[2]        
            # Input projections
            q_ = self.q_post_act_fake_quantize(self.q_proj(q))
            k_ = self.k_post_act_fake_quantize(self.k_proj(k))
            v_ = self.v_post_act_fake_quantize(self.v_proj(v))

            # Separate into heads
            q = self._separate_heads(q_, self.num_heads)
            k = self._separate_heads(k_, self.num_heads)
            v = self._separate_heads(v_, self.num_heads)
            
            # Attention
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn / math.sqrt(c_per_head)
            QK = attn.clone()
            attn = torch.softmax(attn, dim=-1)
            
            attn = self.softmax_post_act_fake_quantize(attn,value=v)

            # Get output
            out = attn @ v
            
            out = self._recombine_heads(out)
            # print(out.shape)
            out = self.out_proj(out)

            return out, (QK, q_.clone(), k_.clone(), v_.clone())
    
    def bimodal_adjust(self):
        if self.k_post_act_fake_quantize.is_bimodal:
            sign = self.k_post_act_fake_quantize.sign
            def addjust_linear(linear:torch.nn.Linear, sign):
                linear.weight.mul_(sign.unsqueeze(1))
                linear.bias.mul_(sign)
            addjust_linear(self.k_proj.module, sign)
            addjust_linear(self.q_proj.module, sign)
            self.k_post_act_fake_quantize.is_bimodal = False


class QuantDecoderOurAttentionBlock_ori(QuantizedBlock):
    def __init__(self, org_module: DecoderAttention, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True, qinput=False):
        super().__init__()
        self.qoutput = qoutput
        self.embedding_dim = org_module.embedding_dim
        self.internal_dim = org_module.internal_dim
        self.num_heads = org_module.num_heads
        
        self.q_proj = PreQuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig)
        self.k_proj = PreQuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig)
        self.v_proj = PreQuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig)
        self.out_proj = PreQuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig)
        # self.out_proj = QuantizedLayer(org_module.out_proj, None, w_qconfig, a_qconfig, qoutput=False)

        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax')
        else:
            softmax_a_config = a_qconfig
        if ptq4sam_config.BIG:
            sign_a_config = update_specialized_quantizer_config(a_qconfig, 'bimodal')
        else:
            sign_a_config = a_qconfig
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)
        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, sign_a_config)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)

        if ptq4sam_config.BIG:
            self.k_post_act_fake_quantize.global_num = ptq4sam_config.global_num
            self.k_post_act_fake_quantize.peak_distance = ptq4sam_config.peak_distance
            self.k_post_act_fake_quantize.peak_height = ptq4sam_config.peak_height

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    # def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    def forward(self, qkv: tuple) -> Tensor:

        q,k,v = qkv[0],qkv[1],qkv[2]
        
        # Input projections
        q = self.q_post_act_fake_quantize(self.q_proj(q))
        k = self.k_post_act_fake_quantize(self.k_proj(k))
        v = self.v_post_act_fake_quantize(self.v_proj(v))
        # print(q.shape, k.shape, v.shape)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        # print(q.shape, k.shape, v.shape)
        
        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        QK = attn.detach().cpu()
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        
        attn = self.softmax_post_act_fake_quantize(attn,value=v)

        # Get output
        out = attn @ v
        
        out = self._recombine_heads(out)
        # print(out.shape)
        out = self.out_proj(out)

        return out, (QK, q.detach(), k.detach(), v.detach())
    
    def bimodal_adjust(self):
        if self.k_post_act_fake_quantize.is_bimodal:
            sign = self.k_post_act_fake_quantize.sign
            def addjust_linear(linear:torch.nn.Linear, sign):
                linear.weight.mul_(sign.unsqueeze(1))
                linear.bias.mul_(sign)
            addjust_linear(self.k_proj.module, sign)
            addjust_linear(self.q_proj.module, sign)
            self.k_post_act_fake_quantize.is_bimodal = False

class Stage(QuantizedTransformerStage):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class QuantImageEncoderOurViT(nn.Module):
    
    def __init__(self, org_module: ImageEncoderViT, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.img_size = org_module.img_size
        self.patch_embed = org_module.patch_embed
        # do not quantize the first block/layer pos_embed
        self.pos_embed = org_module.pos_embed
        
        # stage partition
        self.stages = nn.ModuleList()
        
        block_subset = []
        for i in range(len(org_module.blocks)):
            if org_module.blocks[i].window_size>0:
                block_subset.append(QunatEncoderOurBlock(org_module.blocks[i], w_qconfig, a_qconfig, ptq4sam_config))
            else:
                block_subset.append(QunatEncoderOurBlock(org_module.blocks[i], w_qconfig, a_qconfig, ptq4sam_config))
                self.stages.append(Stage(block_subset))
                block_subset.clear()
                

        self.neck = QuantNeck(org_module.neck, w_qconfig, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, stage in enumerate(self.stages):
            x = stage(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x


# 用的
class QuantImageEncoderOurViT_ori(nn.Module):
    
    def __init__(self, org_module: ImageEncoderViT, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.img_size = org_module.img_size
        self.patch_embed = org_module.patch_embed
        # do not quantize the first block/layer pos_embed
        self.pos_embed = org_module.pos_embed
        
        self.blocks = nn.ModuleList()
        for i in range(len(org_module.blocks)):
            self.blocks.append(QunatEncoderOurBlock(org_module.blocks[i], w_qconfig, a_qconfig, ptq4sam_config))

        self.neck = QuantNeck(org_module.neck, w_qconfig, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

class QunatEncoderOurBlock(QuantizedTransformerLayer):
    def __init__(self, org_module: Block, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True ) -> None:
        super().__init__()
        self.norm1 = org_module.norm1
        self.attn = QuantEncoderOurAttentionBlock(org_module.attn, w_qconfig, a_qconfig, ptq4sam_config)
        # print(self.attn)
        self.norm2 = org_module.norm2
        self.mlp = QuantMLPBlock(org_module.mlp, w_qconfig, a_qconfig)

        self.window_size = org_module.window_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x,_ = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            # print(self.window_size, pad_hw, (H, W))
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class QuantEncoderOurAttentionBlock(QuantizedBlock):
    def __init__(self, org_module: EncoderAttention, w_qconfig, a_qconfig, ptq4sam_config, qoutput=False, qinput=True):
        super().__init__()
        self.qinput = qinput
        self.qoutput = qoutput
        self.num_heads = org_module.num_heads
        self.scale = org_module.scale

        self.qkv = PreQuantizedLayer(org_module.qkv, None, w_qconfig, a_qconfig)
        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig)
        self.use_rel_pos = org_module.use_rel_pos

        if self.use_rel_pos:
            self.rel_pos_h = org_module.rel_pos_h
            self.rel_pos_w = org_module.rel_pos_w
        
        if ptq4sam_config.AGQ:
            softmax_a_config = update_specialized_quantizer_config(a_qconfig,'softmax')
        else:
            softmax_a_config = a_qconfig
        
        self.softmax_post_act_fake_quantize = Quantizer(None, softmax_a_config)

        self.q_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.k_post_act_fake_quantize = Quantizer(None, a_qconfig)
        self.v_post_act_fake_quantize = Quantizer(None, a_qconfig)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        q_ = self.q_post_act_fake_quantize(q)
        k_ = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        attn = (q_ * self.scale) @ k_.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = self.softmax_post_act_fake_quantize(attn.softmax(dim=-1), value=v)
        
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        
        return x, (q_.clone(), k_.clone())

specials = {
    TwoWayAttentionBlock: QuantDecoderOurTwoWayAttentionBlock,
    Attention: QuantDecoderOurAttentionBlock,
    ImageEncoderViT: QuantImageEncoderOurViT,
}

def bimodal_adjust(model,logger):
    logger.info('start to detect dimodal distribution')
    for name,m in model.named_modules():
        if isinstance(m, QuantDecoderOurAttentionBlock) and 'token_to_image' in name:
            logger.info(name)
            # print(m.k_post_act_fake_quantize.is_A_two_peak, m.k_post_act_fake_quantize.is_bimodal)
            logger.info(m.k_post_act_fake_quantize.is_bimodal)
            m.bimodal_adjust()
    logger.info('bimodal integration end')

def enable_rank_observe(model, logger):
    logger.info('******enable observe ARR')
    for name,m in model.named_modules():
        if isinstance(m, QuantDecoderOurAttentionBlock):
            m.AttnRank = True
            m.AttnRank_calib = True
            # m.ori_ATM = None
            
def enable_get_ori_ATM(model, logger):
    logger.info('******enable get ori ATM')
    for name,m in model.named_modules():
        if isinstance(m, QuantDecoderOurAttentionBlock):
            m.AttnRank = False
            m.get_ori_ATM = True
    
    