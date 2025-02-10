import torch
import torch.nn as nn
from torch.hub import load
import timm
from ProViCNet.ModelArchitectures.ProViCNet.vit_config import dino_backbones
import torch.nn.init as init
from ProViCNet.ModelArchitectures.lora import _LoRA_qkv_timm
from ProViCNet.ModelArchitectures.base_vit import ViT
import math

class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class conv_head8(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(conv_head8, self).__init__()
        # Adjusting the architecture to include three upsampling layers
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),  # First upsampling
            nn.Conv2d(embedding_size, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Second upsampling
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Third upsampling
            nn.Conv2d(64, num_classes, 3, padding=1),
            #nn.Sigmoid()  # Using Sigmoid for the final layer
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        return x

class conv_head14(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(conv_head, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=7),
            nn.Conv2d(embedding_size, 64, (3,3), padding=(1,1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3,3), padding=(1,1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)
        return x

class conv_head8_3up(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(conv_head8_3up, self).__init__()
        # Improved architecture with additional convolution layers and skip connections
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )

    def forward(self, x):
        # Forward pass with skip connections
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        return x3

class ProViCNet(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones, n_modal=4):
        super(ProViCNet, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2': conv_head14,
            'conv3': conv_head8_3up
        }
        self.num_classes =  num_classes # add a class for background if needed
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        
        self.additional_layers = 2
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        x = torch.cat(
            (
                x[:, :1], # 1
                self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                axis_pos, # 1
                axis_max, # 1
                x[:, 1:],
            ),
            dim=1,
        )
        return x
    
    def forward_features_pos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")
        
        x = self.prepare_tokens_with_axispos(x, pos)
        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # 0
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],  # 1:5
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens + 1 + self.additional_layers ], # 5:8 (cls, regist4, other 3)
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 1 + self.additional_layers :],
            "x_prenorm": x,
        }
    
    def forward(self, x, pos=None, modal=None): # pos should be (axis_pos, axis_max)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if modal is not None:
            x = self.forward_features_pos(x, pos, modal)
        elif pos is not None:
            x = self.forward_features_pos(x, pos)
        else:
            x = self.backbone.forward_features(x)
        x = x['x_norm_patchtokens'] # (batch_size, num_patches, embeddings)
        # x_norm_clstoken # [1, 384]
        # x_norm_regtokens # [1, 4, 384]
        # x_norm_postokens # [1, 2, 384] if pos available
        # x_norm_patchtokens # [1, 1024, 384]

        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        x = x.reshape(batch_size, self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) # (batch_size, embeddings, y-patch, x-patch)
        x = self.head(x) # (batch_size, num_classes, label-y, label-x)
        return x

class ProViCNet_modal(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones, n_modal=4):
        super(ProViCNet_modal, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2': conv_head14,
            'conv3': conv_head8_3up
        }
        self.num_classes =  num_classes # add a class for background if needed
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        
        self.additional_layers = 3
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        self.modal = nn.Linear(n_modal, self.embedding_size)
        
        
        self.init_weights()

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)

        init.normal_(self.modal.weight, mean=0.0, std=0.02)
        init.constant_(self.modal.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        if modal is not None:
            modality = self.modal(modal).unsqueeze(1)         # [batch, embedding] -> [batch, 1, embedding]
            x = torch.cat(
                (
                    x[:, :1], # 1
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                    axis_pos, # 1
                    axis_max, # 1
                    modality, # 1
                    x[:, 1:],
                ),
                dim=1,
            )
        else:
            x = torch.cat(
                (
                    x[:, :1], # 1
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                    axis_pos, # 1
                    axis_max, # 1
                    x[:, 1:],
                ),
                dim=1,
            )
        return x
    
    def forward_features_pos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")

        if modal is not None:
            x = self.prepare_tokens_with_axispos(x, pos, modal)
            additional_layers = 3
        else:
            x = self.prepare_tokens_with_axispos(x, pos)
            additional_layers = 2

        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # 0
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],  # 1:5
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens + 1 + additional_layers ], # 5:8 (cls, regist4, other 3)
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 1 + additional_layers :],
            "x_prenorm": x,
        }
    
    def forward(self, x, pos=None, modal=None): # pos should be (axis_pos, axis_max)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if modal is not None:
            x = self.forward_features_pos(x, pos, modal)
        elif pos is not None:
            x = self.forward_features_pos(x, pos)
        else:
            x = self.backbone.forward_features(x)
        x = x['x_norm_patchtokens'] # (batch_size, num_patches, embeddings)
        # x_norm_clstoken # [1, 384]
        # x_norm_regtokens # [1, 4, 384]
        # x_norm_postokens # [1, 2, 384] if pos available
        # x_norm_patchtokens # [1, 1024, 384]

        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        x = x.reshape(batch_size, self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) # (batch_size, embeddings, y-patch, x-patch)
        x = self.head(x) # (batch_size, num_classes, label-y, label-x)
        return x

class ProViCNet_LoRA(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones, n_modal=4,
                       r = 4, alpha = 4, freeze = True):
        super(ProViCNet_LoRA, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2': conv_head14,
            'conv3': conv_head8_3up
        }
        self.num_classes =  num_classes # add a class for background if needed
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']


        self.backbone_dino = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.lora_layer = list(range(len(self.backbone_dino.blocks)))
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        if freeze:
            for param in self.backbone_dino.parameters():
                param.requires_grad = False
                
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(self.backbone_dino.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                r,
                alpha
            )
        self.reset_parameters()
        self.backbone = self.backbone_dino
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        
        self.additional_layers = 3
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        self.modal = nn.Linear(n_modal, self.embedding_size)
        
        
        self.init_weights()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)

        init.normal_(self.modal.weight, mean=0.0, std=0.02)
        init.constant_(self.modal.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        if modal is not None:
            modality = self.modal(modal).unsqueeze(1)         # [batch, embedding] -> [batch, 1, embedding]
            x = torch.cat(
                (
                    x[:, :1], # 1
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                    axis_pos, # 1
                    axis_max, # 1
                    modality, # 1
                    x[:, 1:],
                ),
                dim=1,
            )
        else:
            x = torch.cat(
                (
                    x[:, :1], # 1
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                    axis_pos, # 1
                    axis_max, # 1
                    x[:, 1:],
                ),
                dim=1,
            )
        return x
    
    def forward_features_pos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")

        if modal is not None:
            x = self.prepare_tokens_with_axispos(x, pos, modal)
            additional_layers = 3
        else:
            x = self.prepare_tokens_with_axispos(x, pos)
            additional_layers = 2

        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # 0
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],  # 1:5
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens + 1 + additional_layers ], # 5:8 (cls, regist4, other 3)
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 1 + additional_layers :],
            "x_prenorm": x,
        }
    
    def forward(self, x, pos=None, modal=None): # pos should be (axis_pos, axis_max)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if modal is not None:
            x = self.forward_features_pos(x, pos, modal)
        elif pos is not None:
            x = self.forward_features_pos(x, pos)
        else:
            x = self.backbone.forward_features(x)
        x = x['x_norm_patchtokens'] # (batch_size, num_patches, embeddings)
        # x_norm_clstoken # [1, 384]
        # x_norm_regtokens # [1, 4, 384]
        # x_norm_postokens # [1, 2, 384] if pos available
        # x_norm_patchtokens # [1, 1024, 384]

        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        x = x.reshape(batch_size, self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) # (batch_size, embeddings, y-patch, x-patch)
        x = self.head(x) # (batch_size, num_classes, label-y, label-x)
        return x

class ContrastiveLearning_Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class ProViCNet_contrastive_US(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones, n_modal=4, in_channels=19):
        super(ProViCNet_contrastive_US, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2': conv_head14,
            'conv3': conv_head8_3up
        }
        self.ChannelReducer = ChannelReducer(in_channels=in_channels, out_channels=3)
        self.CL_Head = ContrastiveLearning_Head(
            in_dim=backbones[backbone]['embedding_size'],
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True
        )

        self.num_classes =  num_classes # add a class for background if needed
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        
        self.additional_layers = 2
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        x = torch.cat(
            (
                x[:, :1], # 1
                self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                axis_pos, # 1
                axis_max, # 1
                x[:, 1:],
            ),
            dim=1,
        )
        return x
    
    def forward_features_pos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")
        
        x = self.prepare_tokens_with_axispos(x, pos)
        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # 0
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],  # 1:5
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens + 1 + self.additional_layers ], # 5:8 (cls, regist4, other 3)
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 1 + self.additional_layers :],
            "x_prenorm": x,
        }
    
    def forward(self, x, pos=None, modal=None, return_features=False, forward_head=False): # pos should be (axis_pos, axis_max)
        if forward_head:
            head_module = getattr(self, 'module', self).CL_Head
            return head_module(x)
        
        x = self.ChannelReducer(x) #
        # preds = MODEL(Image, pos)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if modal is not None:
            x = self.forward_features_pos(x, pos, modal)
        elif pos is not None:
            x = self.forward_features_pos(x, pos)
        else:
            x = self.backbone.forward_features(x)
        x = x['x_norm_patchtokens'] # (batch_size, num_patches, embeddings)
        # x_norm_clstoken # [1, 384]
        # x_norm_regtokens # [1, 4, 384]
        # x_norm_postokens # [1, 2, 384] if pos available
        # x_norm_patchtokens # [1, 1024, 384]

        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        features = x.reshape(batch_size, self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) # (batch_size, embeddings, y-patch, x-patch)
        x = self.head(features) # (batch_size, num_classes, label-y, label-x)
        if return_features:
            return x, features
        return x



class ProViCNet_contrastive(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones, n_modal=4):
        super(ProViCNet_contrastive, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2': conv_head14,
            'conv3': conv_head8_3up
        }

        self.CL_Head = ContrastiveLearning_Head(
            in_dim=384,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True
        )

        self.num_classes =  num_classes # add a class for background if needed
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        
        self.additional_layers = 2
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        x = torch.cat(
            (
                x[:, :1], # 1
                self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                axis_pos, # 1
                axis_max, # 1
                x[:, 1:],
            ),
            dim=1,
        )
        return x
    
    def forward_features_pos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")
        
        x = self.prepare_tokens_with_axispos(x, pos)
        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # 0
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],  # 1:5
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens + 1 + self.additional_layers ], # 5:8 (cls, regist4, other 3)
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 1 + self.additional_layers :],
            "x_prenorm": x,
        }
    
    def forward(self, x, pos=None, modal=None, return_features=False, forward_head=False): # pos should be (axis_pos, axis_max)
        if forward_head:
            head_module = getattr(self, 'module', self).CL_Head
            return head_module(x)
        # preds = MODEL(Image, pos)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if modal is not None:
            x = self.forward_features_pos(x, pos, modal)
        elif pos is not None:
            x = self.forward_features_pos(x, pos)
        else:
            x = self.backbone.forward_features(x)
        x = x['x_norm_patchtokens'] # (batch_size, num_patches, embeddings)
        # x_norm_clstoken # [1, 384]
        # x_norm_regtokens # [1, 4, 384]
        # x_norm_postokens # [1, 2, 384] if pos available
        # x_norm_patchtokens # [1, 1024, 384]

        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        features = x.reshape(batch_size, self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) # (batch_size, embeddings, y-patch, x-patch)
        x = self.head(features) # (batch_size, num_classes, label-y, label-x)
        if return_features:
            return x, features
        return x

class ProViCNet_contrastive_modal(nn.Module): #
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones, n_modal=4):
        super(ProViCNet_contrastive_modal, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2': conv_head14,
            'conv3': conv_head8_3up
        }

        self.CL_Head = ContrastiveLearning_Head(
            in_dim=384,
            out_dim=65536,
            use_bn=False,
            norm_last_layer=True
        )

        self.num_classes =  num_classes # add a class for background if needed
        self.backbones = dino_backbones
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']

        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        
        self.additional_layers = 3
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        self.modal = nn.Linear(n_modal, self.embedding_size)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)
        init.normal_(self.modal.weight, mean=0.0, std=0.02)
        init.constant_(self.modal.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)
        
        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        modality = self.modal(modal).unsqueeze(1)

        x = torch.cat(
            (
                x[:, :1], # 1
                self.backbone.register_tokens.expand(x.shape[0], -1, -1),  # 4
                axis_pos, # 1
                axis_max, # 1
                modality,
                x[:, 1:],
            ),
            dim=1,
        )
        return x
    
    def forward_features_pos(self, x, pos, modal=None): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")
        
        x = self.prepare_tokens_with_axispos(x, pos, modal)
        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],  # 0
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],  # 1:5
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens + 1 + self.additional_layers ], # 5:8 (cls, regist4, other 3)
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 1 + self.additional_layers :],
            "x_prenorm": x,
        }

    
    def forward(self, x, pos=None, modal=None, return_features=False, forward_head=False): # pos should be (axis_pos, axis_max)
        if forward_head:
            head_module = getattr(self, 'module', self).CL_Head
            return head_module(x)
        
        # preds = MODEL(Image, pos)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if modal is not None:
            x = self.forward_features_pos(x, pos, modal)
        elif pos is not None:
            x = self.forward_features_pos(x, pos)
        else:
            x = self.backbone.forward_features(x)
        x = x['x_norm_patchtokens'] # (batch_size, num_patches, embeddings)
        # x_norm_clstoken # [1, 384]
        # x_norm_regtokens # [1, 4, 384]
        # x_norm_postokens # [1, 2, 384] if pos available
        # x_norm_patchtokens # [1, 1024, 384]

        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        features = x.reshape(batch_size, self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) # (batch_size, embeddings, y-patch, x-patch)
        x = self.head(features) # (batch_size, num_classes, label-y, label-x)
        if return_features:
            return x, features
        return x


class ChannelReducer(nn.Module):
    def __init__(self, in_channels=19, out_channels=3):
        super(ChannelReducer, self).__init__()
        self.reduce_channels = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        x = self.reduce_channels(x)
        return x
    
    
class ProViNet(nn.Module):
    def __init__(self, num_classes, head = 'conv1'):
        super(ProViNet, self).__init__()
        self.heads = {
            'conv1': conv_head8,
            'conv2':conv_head14,
            'conv3': conv_head8_3up
        }
        self.num_classes =  num_classes # add a class for background if needed
        self.backbone = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True)
        self.embedding_size = 768
        self.patch_size = 14

        self.head = self.heads[head](self.embedding_size,self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        
        # skip cls token
        x = self.backbone.forward_features(x)[:, 1:] # (batch_size, num_patches, embeddings)
        
        x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1])) 
        # (batch_size, embeddings, y-patch, x-patch)

        # Segmentation part
        x = self.head(x)
        return x

# Simple MODEL_Fusion
class ModalitySpecificStream(nn.Module):
    def __init__(self, embedding_size=384, target_channels=128):
        super(ModalitySpecificStream, self).__init__()
        self.UpsampleModule = nn.Sequential(
            nn.Upsample(scale_factor=2),  # First upsampling
            nn.Conv2d(embedding_size, target_channels, 3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.UpsampleModule(x)
        return x

class FusionModalities(nn.Module):
    def __init__(self, embedding_size=384, target_channels=128, num_classes=3):
        super(FusionModalities, self).__init__()
        self.embedding_size = embedding_size
        self.Stream_T2W = ModalitySpecificStream(embedding_size)
        self.Stream_ADC = ModalitySpecificStream(embedding_size)
        self.Stream_DWI = ModalitySpecificStream(embedding_size)
        
        self.FusionSegmentation = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Second upsampling
            nn.Conv2d(target_channels*3, target_channels, 3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Third upsampling
            nn.Conv2d(target_channels, num_classes, 3, padding=1),
        )

    def change_input_dim(self, x):
        batch_size = x.shape[0]
        mask_dim = x.shape[1] ** 0.5
        x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
        x = x.reshape(batch_size, self.embedding_size, int(mask_dim), int(mask_dim)) # (batch_size, embeddings, y-patch, x-patch)
        return x

    def forward(self, token_T2W, token_ADC, token_DWI):
        token_T2W, token_ADC, token_DWI = (self.change_input_dim(token) for token in (token_T2W, token_ADC, token_DWI))
        T2W_map = self.Stream_T2W(token_T2W)
        ADC_map = self.Stream_ADC(token_ADC)
        DWI_map = self.Stream_DWI(token_DWI)
        Fusion_map = torch.cat((T2W_map, ADC_map, DWI_map), dim=1)
        Fusion_map = self.FusionSegmentation(Fusion_map)
        return Fusion_map
    

# Complex MODEL_Fusion
# class ModalitySpecificStream(nn.Module):
#     def __init__(self, embedding_size=384, target_channels=128):
#         super(ModalitySpecificStream, self).__init__()
        
#         self.Block1 = nn.Sequential(
#             nn.Conv2d(embedding_size, embedding_size, 3, padding=1, groups=embedding_size), # Depthwise Convolution
#             nn.Conv2d(embedding_size, target_channels, 1),  #
#             nn.BatchNorm2d(target_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.Block1Connection = nn.Conv2d(embedding_size + target_channels, embedding_size, 1)
#         self.UpsampleModule = nn.Sequential(
#             nn.Upsample(scale_factor=2),  # First upsampling
#             nn.Conv2d(embedding_size, target_channels, 3, padding=1),
#             nn.BatchNorm2d(target_channels),
#             nn.ReLU()
#         )
#     def forward(self, x):
#         # Skip Connection의 시작점
#         identity = x
#         x = self.Block1(x)
#         x = torch.cat((identity, x), dim=1)
#         x = self.Block1Connection(x)
#         x = self.UpsampleModule(x)
#         return x

# class FusionModalities(nn.Module):
#     def __init__(self, embedding_size=384, target_channels=128, num_classes=3):
#         super(FusionModalities, self).__init__()
#         self.embedding_size = embedding_size
#         self.Stream_T2W = ModalitySpecificStream(embedding_size)
#         self.Stream_ADC = ModalitySpecificStream(embedding_size)
#         self.Stream_DWI = ModalitySpecificStream(embedding_size)
        
#         self.Block1 = nn.Sequential(
#             nn.Conv2d(target_channels*3, target_channels*3, 3, padding=1, groups=target_channels*3), # Depthwise Separable Convolution
#             nn.Conv2d(target_channels*3, target_channels, 1),  # Pointwise Convolution
#             nn.BatchNorm2d(target_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.Block1Connection = nn.Conv2d(target_channels*3 + target_channels, target_channels*3, 1)

#         self.FusionSegmentation = nn.Sequential(
#             nn.Upsample(scale_factor=2),  # Second upsampling
#             nn.Conv2d(target_channels*3, target_channels, 3, padding=1),
#             nn.BatchNorm2d(target_channels),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),  # Third upsampling
#             nn.Conv2d(target_channels, num_classes, 3, padding=1),
#         )

#     def change_input_dim(self, x):
#         batch_size = x.shape[0]
#         mask_dim = x.shape[1] ** 0.5
#         x = x.permute(0,2,1) # (batch_size, embeddings, num_patches)
#         x = x.reshape(batch_size, self.embedding_size, int(mask_dim), int(mask_dim)) # (batch_size, embeddings, y-patch, x-patch)
#         return x

#     def forward(self, token_T2W, token_ADC, token_DWI):
#         token_T2W, token_ADC, token_DWI = (self.change_input_dim(token) for token in (token_T2W, token_ADC, token_DWI))
#         T2W_map = self.Stream_T2W(token_T2W)
#         ADC_map = self.Stream_ADC(token_ADC)
#         DWI_map = self.Stream_DWI(token_DWI)
#         Fusion_map = torch.cat((T2W_map, ADC_map, DWI_map), dim=1)

#         identity = Fusion_map
#         x = self.Block1(Fusion_map)
#         x = torch.cat((identity, x), dim=1)
#         x = self.Block1Connection(x)
#         Fusion_map = self.FusionSegmentation(x)
#         return Fusion_map
    
# x = torch.rand(30, 384, 32, 32)
# self = FusionModalities()

def load_partial_weights(source_model, target_stream):
    # Extract the weights corresponding to the first 4 layers of the source model
    target_state_dict = target_stream.state_dict()
    source_state_dict = {k: v for idx, (k, v) in enumerate(source_model.state_dict().items()) if idx < len(target_state_dict.keys())}
    
    updated_state_dict = {}
    for key in target_state_dict.keys():
        if key in source_state_dict:
            updated_state_dict[key] = source_state_dict[key]
    
    target_stream.load_state_dict(updated_state_dict, strict=True)


if __name__ == "__main__":
    x = torch.rand(1, 3, 448, 448)

    model_dino = ProViCNet(num_classes=3, head='conv1')
    model_vit  = ProViNet(num_classes=3, head='conv1')

    model_dino = model_dino.cuda()
    model_vit  = model_vit.cuda()
    
    res_dino = model_dino(x.cuda())
    res_vit  = model_dino(x.cuda())


