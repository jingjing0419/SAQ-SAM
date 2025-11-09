from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

import math
import sys
sys.path.append("..")
# from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import Attention, add_decomposed_rel_pos
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from projects.instance_segment_anything.models.segment_anything.modeling.image_encoder import Attention as Attention_Encoder
from projects.instance_segment_anything.models.segment_anything.modeling.transformer import Attention as Attention_Decoder





def attention_forward_encoder(self, x):
    # print(x.shape)      # torch.Size([4, 14, 14, 768])  # torch.Size([1, 16, 16, 768])
    # print('build attention forward ... ')
    B, H, W, _ = x.shape
    # qkv with shape (3, B, nHead, H * W, C)
    # print(self.qkv.weight.shape)    #torch.Size([2304, 768])
    # print(self.qkv(x).shape)        # torch.Size([4, 14, 14, 2304])
    # exit()
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    # q, k, v with shape (B * nHead, H * W, C)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

    # attn = (q * self.scale) @ k.transpose(-2, -1)
    # attn = self.matmul1(q * self.scale, k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2, -1))* self.scale

    if self.use_rel_pos:
        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

    attn = attn.softmax(dim=-1)
    # x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    x = self.matmul2(attn, v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    x = self.proj(x)

    return x

def attention_forward_decoder(self, q: Tensor, k: Tensor, v: Tensor):
    # Input projections
    # print(q.size(),k.size(),v.size())
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    # Attention
    _, _, _, c_per_head = q.shape
    # attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
    attn = self.matmul1(q, k.permute(0, 1, 3, 2))
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)

    # Get output
    # out = attn @ v
    out = self.matmul2(attn, v)
    out = self._recombine_heads(out)
    out = self.out_proj(out)
    # print('forward...')
    return out



def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(model):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.
    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384
    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    # model = timm.create_model(name, pretrained=True)

    #for module in model.modules():
    for name, module in model.named_modules():
        if isinstance(module, Attention_Encoder):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward_encoder, module)
        if isinstance(module, Attention_Decoder):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward_decoder, module)
        # if isinstance(module, WindowAttention):
        #     setattr(module, "matmul1", MatMul())
        #     setattr(module, "matmul2", MatMul())
        #     module.forward = MethodType(window_attention_forward, module)

    # return model

def adapter_channel_wise(args, q_model):
    for module in q_model.image_encoder.blocks:
        module.Adapter.channel[0].input_quantizer.channel_wise = True
        module.Adapter.channel[0].forward = MethodType(adapter_quantizer_forward, module.Adapter.channel[0])
        
    

def adapter_quantizer_forward(self, x):
    if self.use_input_quant:
        # if self.BHWC_Act:               # 激活(B, H, W, C)变为（B, H*W, C）来量化，然后再变回去
        #     B, H, W, C = x.shape
        #     # print('ok')
        #     x = x.view(B, H*W, C)
        #     x = self.input_quantizer(x)
        #     x = x.view(B, H, W, C)
        #     # print(x.shape)
        # else:
        # print(x.shape)
        # exit()
        
        B, C = x.shape
        x = x.view(B, 1, C)
        print(x.shape)
        x = self.input_quantizer(x)
        x = x.view(B, C)
    # else:
    #     print('no act quant!')
            
    if self.use_weight_quant:
        w = self.weight_quantizer(self.weight)
    else:
        w = self.weight
        # print('no weight quant!')

    out = F.linear(x, weight=w, bias=self.bias)

    return out