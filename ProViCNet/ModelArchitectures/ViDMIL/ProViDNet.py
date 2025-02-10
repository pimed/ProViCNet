import torch
import torch.nn as nn
from torch.hub import load
import timm
from .vit_config import dino_backbones
import torch.nn.init as init

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

class ProViDNet(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s_reg', head = 'conv1', backbones = dino_backbones):
        super(ProViDNet, self).__init__()
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
        self.axis_pos = nn.Linear(1, self.embedding_size)
        self.axis_max = nn.Linear(1, self.embedding_size)
        
        self.init_weights()

    def init_weights(self):
        init.normal_(self.axis_pos.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_pos.bias, 0.0)
        
        init.normal_(self.axis_max.weight, mean=0.0, std=0.02)
        init.constant_(self.axis_max.bias, 0.0)

    def prepare_tokens_with_axispos(self, x, pos): # pos should be (axis_pos, axis_max), shape: [batch_size, 2]
        B, nc, w, h = x.shape
        x = self.backbone.patch_embed(x)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        axis_pos = self.axis_pos(pos[:,0:1]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]
        axis_max = self.axis_max(pos[:,1:2]).unsqueeze(1) # [batch, embedding] -> [batch, 1, embedding]

        if self.backbone.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),
                    axis_pos,
                    axis_max,
                    x[:, 1:],
                ),
                dim=1,
            )
        return x
    def forward_features_pos(self, x, pos): # pos should be (axis_pos, axis_max)
        if isinstance(x, list):
            #return self.forward_features_list(x, masks)
            raise ValueError("The input data type is incorrect. Tensor type input is required.")

        x = self.prepare_tokens_with_axispos(x, pos)

        for blk in self.backbone.blocks:
            x = blk(x)

        x_norm = self.backbone.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.backbone.num_register_tokens + 1],
            "x_norm_postokens": x_norm[:, self.backbone.num_register_tokens+1: self.backbone.num_register_tokens+3 ],
            "x_norm_patchtokens": x_norm[:, self.backbone.num_register_tokens + 3 :],
            "x_prenorm": x,
        }
    
    def forward(self, x, pos=None): # pos should be (axis_pos, axis_max)
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        if pos is not None:
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
    
if __name__ == "__main__":
    x = torch.rand(1, 3, 448, 448)

    model_dino = ProViDNet(num_classes=3, head='conv1')
    model_vit  = ProViNet(num_classes=3, head='conv1')

    model_dino = model_dino.cuda()
    model_vit  = model_vit.cuda()
    
    res_dino = model_dino(x.cuda())
    res_vit  = model_dino(x.cuda())


