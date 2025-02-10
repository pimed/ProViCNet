# run this file ./
import sys
import torch
import torch.nn as nn
import timm

def GetModel(ModelName, nClass, nChannel, img_size, vit_backbone='dinov2_s_reg', modal=False, contrastive=False, freeze=True, US=False):
    if ModelName == "ProViCNet":
        assert img_size == 448, "Image size must be 448 for ProViCNet"
        #assert nChannel == 3, "Number of channels must be 3 for ProViNet"
        from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import ProViCNet, ProViCNet_modal, ProViCNet_contrastive, ProViCNet_contrastive_modal, ProViCNet_contrastive_US
        if modal and contrastive:
            print("MODEL: ProViCNet_contrastive_modal called") 
            MODEL = ProViCNet_contrastive_modal(num_classes=nClass, backbone=vit_backbone, head='conv1')
        elif modal:
            print("MODEL: ProViCNet_modal called")
            MODEL = ProViCNet_modal(num_classes=nClass, backbone=vit_backbone, head='conv1')
        elif US and contrastive:
            print("MODEL: Ultrasound ProViCNet_contrastive called")
            MODEL = ProViCNet_contrastive_US(num_classes=nClass, backbone=vit_backbone, head='conv1', in_channels=nChannel)
        elif contrastive:
            print("MODEL: ProViCNet_contrastive called")
            MODEL = ProViCNet_contrastive(num_classes=nClass, backbone=vit_backbone, head='conv1')
        else:
            print("MODEL: ProViCNet called")
            MODEL = ProViCNet(num_classes=nClass, backbone=vit_backbone, head='conv1')
    elif ModelName == "ProViNet":
        assert img_size == 448, "Image size must be 448 for ProViNet"
        assert nChannel == 3, "Number of channels must be 3 for ProViNet"
        from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import ProViNet
        MODEL = ProViNet(num_classes=3, head='conv1')
    elif ModelName == "ProViCNet_LoRA":
        assert img_size == 448, "Image size must be 448 for ProViNet"
        assert nChannel == 3, "Number of channels must be 3 for ProViNet"
        from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import ProViCNet_LoRA
        MODEL = ProViCNet_LoRA(num_classes=3, backbone=vit_backbone, head='conv1', freeze=freeze)
    elif ModelName == "UCTransNet":
        # sys.path.append('/home/sosal/student_projects/JeongHoonLee/TRUS/Code/')
        from ProViCNet.ModelArchitectures.UCTransNet.nets.UCTransNet import UCTransNet
        import ModelArchitectures.UCTransNet.Config as config
        config_vit = config.get_CTranS_config()
        config_vit['n_classes'] = nClass
        MODEL = UCTransNet(config=config_vit, n_channels=nChannel, n_classes=nClass, img_size=img_size, vis=False)
    elif ModelName == "UNet":
        from ProViCNet.ModelArchitectures.unet import UNet
        MODEL = UNet(nClass)
    elif ModelName == "SwinUNet":
        from ProViCNet.ModelArchitectures.SwinUNet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
        MODEL = SwinTransformerSys(img_size=img_size, window_size=8, num_classes=nClass)
    elif ModelName == "MISSFormer":
        from ProViCNet.ModelArchitectures.MISSFormer.MISSFormer import MISSFormer
        MODEL = MISSFormer(num_classes=3)
    elif ModelName == "TransUNet":
        from ProViCNet.ModelArchitectures.TransUNet.transunet import TransUNet
        MODEL = TransUNet(img_dim=img_size, in_channels=nChannel, class_num=nClass)
    elif ModelName == "NestedUNet":
        from ProViCNet.ModelArchitectures.NestedUNet.nestedUnet import NestedUNet
        MODEL = NestedUNet(nClass, input_channels=3)
    elif ModelName == "LeViTUnet":
        from ProViCNet.ModelArchitectures.LeViTUnet.LeViTUnet import Build_LeViT_UNet_384
        MODEL = Build_LeViT_UNet_384(num_classes=nClass, pretrained=False)
    
    return MODEL

if __name__ == "__main__":
    ModelName = 'ProViNet' 

    if ModelName == "ProViCNet":
        MODEL = GetModel("ProViCNet", nClass=3, nChannel=3, img_size=448)
        x = torch.rand(7, 3, 448, 448)
    elif ModelName == "UCTransNet":
        MODEL = GetModel("UCTransNet", nClass=3, nChannel=3, img_size=256)
        x = torch.rand(7, 3, 256, 256)
    elif ModelName == "UNet":
        MODEL = GetModel("UNet", nClass=3, nChannel=3, img_size=256)
        x = torch.rand(7, 3, 256, 256)
    
    MODEL = MODEL.cuda()
    res = MODEL(x.cuda())
    print(f"Shape of the segmentation Output: {res.shape}")