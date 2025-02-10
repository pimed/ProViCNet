import argparse
import json
import os
import yaml
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from ProViCNet.ModelArchitectures.Models import GetModel
from util_functions.Prostate_DataGenerator import US_MRI_Generator, collate_prostate_position_CS, getData
from util_functions.utils_weighted import initialize_weights
from util_functions.train_functions import OneBatchTraining_seg_contrastive, tensor_shuffle

def main(args, config):
    # 1. Data Loading #############################################
    Dataset = getData(config['paths']['Image_path'],
                      config['paths']['Gland_path'],
                      config['paths']['Label_path'],
                      config['Modality'], config['file_extensions'])

    with open(config['SplitValidation']['internal_split'], 'r') as file:
        SplitValidation_dict = json.load(file)
    with open(config['SplitValidation']['testset'], 'r') as file:
        SplitValidation_dict_test = json.load(file)
        SplitValidation_dict_test = pd.DataFrame(SplitValidation_dict_test['bx_test'])
    
    TrainDataset = Dataset.loc[SplitValidation_dict[config['FOLD_IDX']]['train']]
    ValidDataset = Dataset.loc[SplitValidation_dict[config['FOLD_IDX']]['val']]
    TrainDataset.reset_index(drop=True, inplace=True)
    ValidDataset.reset_index(drop=True, inplace=True)

    TRAIN_GENERATOR = US_MRI_Generator(
                    imageFileName=TrainDataset['Image'],
                    glandFileName=TrainDataset['Gland'],
                    cancerFileName=TrainDataset['Cancer'],
                    modality=TrainDataset['Modality'],
                    cancerTo2=False, Augmentation=True,
                    filter_background_prob=args.filter_background_prob, img_size=args.img_size)
                    
    VALID_GENERATOR = US_MRI_Generator(
                    imageFileName=ValidDataset['Image'],
                    glandFileName=ValidDataset['Gland'],
                    cancerFileName=ValidDataset['Cancer'],
                    modality=ValidDataset['Modality'],
                    cancerTo2=False, Augmentation=False,
                    filter_background_prob=args.filter_background_prob_valid, img_size=args.img_size)

    # if position info is not available, then use collate_prostate
    TRAIN_DATALOADER = DataLoader(TRAIN_GENERATOR, batch_size=args.Bag_batch_size, shuffle=True,
                                  collate_fn=collate_prostate_position_CS, num_workers=args.num_workers)
    VALID_DATALOADER = DataLoader(VALID_GENERATOR, batch_size=1, shuffle=False,
                                  collate_fn=collate_prostate_position_CS, num_workers=args.num_workers)

    # 2. Model Loading ############################################
    print("# 2. DINO Segmentation Model Loading ############## ") #
    MODEL = GetModel(args.ModelName, args.nClass, args.nChannel, args.img_size, vit_backbone=args.vit_backbone, contrastive=args.contrastive, US=args.US)
    if args.ModelName in ('ProViDNet', 'ProViCNet'):
        print(f"ViDMIL: DINO backbone with low Learning Rate: {args.learning_rate * args.DINO_learning_rate}/{args.learning_rate}")
        initialize_weights(MODEL.head)
        if args.contrastive:
            initialize_weights(MODEL.CL_Head)
        backbone_params = list(MODEL.backbone.parameters()) # list(MODEL.module.backbone.parameters())
        head_params = list(MODEL.head.parameters()) # list(MODEL.module.head.parameters())
        axis_pos_params = list(MODEL.axis_pos.parameters())
        axis_max_params = list(MODEL.axis_max.parameters())
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': args.learning_rate * args.DINO_learning_rate},
            {'params': head_params + axis_pos_params + axis_max_params, 'lr': args.learning_rate}
        ])
    else:
        initialize_weights(MODEL)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, MODEL.parameters()), lr=args.learning_rate)  # Choose optimize
    MODEL = MODEL.to(args.device)
    
    if args.pretrained_weights:
        MODEL.load_state_dict(torch.load(args.pretrained_weights, map_location=args.device), strict=True)
        
    weights_trn = torch.tensor(args.train_class_weight, device=args.device)
    weights_val = torch.tensor(args.valid_class_weight, device=args.device)
    criterion_trn = DiceCELoss(to_onehot_y=True, softmax=True, weight=weights_trn)
    criterion_val = DiceCELoss(to_onehot_y=True, softmax=True, weight=weights_val)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, min_lr=1e-6)
    os.makedirs(args.save_directory, exist_ok=True)

    # 3. Training & Validation & Test #############################
    for epoch in range(1, args.epoch, 1):
        trn_loss, trn_count = 0, 0
        # Training
        MODEL.train()
        for bag_batch_idx, (Images, Labels, Positions) in tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER)):
            Images, Labels, Positions = tensor_shuffle(Images, Labels, args.device, pos = Positions, shuffle=True)
            for idx in range(0, Images.shape[0], args.small_batchsize):
                Image, Label, Posit = Images[idx:idx+args.small_batchsize, :, :, :], \
                                      Labels[idx:idx+args.small_batchsize, :, :, :], \
                                      Positions[idx:idx+args.small_batchsize, :]
                if Label[:,2:4].sum() == 0: continue
                loss = OneBatchTraining_seg_contrastive(Image, Label, MODEL, criterion_trn, optimizer, pos = Posit,
                                                        contrastive_alpha=args.contrastive_alpha,
                                                        max_contrastive_pairs = args.max_pair, distant_negative=True)
                trn_loss += loss
                trn_count += 1
        trn_loss /= trn_count

        # Validation set
        MODEL.eval()
        val_loss, val_count = 0, 0
        for bag_batch_idx, (Images, Labels, Positions) in tqdm(enumerate(VALID_DATALOADER), total=len(VALID_DATALOADER)):
            Images, Labels, Positions = tensor_shuffle(Images, Labels, device=args.device, pos=Positions, shuffle=False)
            with torch.no_grad():
                pred = MODEL(Images, Positions)
                Label_argmax = Labels.argmax(axis=1).unsqueeze(1)
                val_loss += criterion_val(pred, Label_argmax)
                val_count += 1
        val_loss /= val_count
        saveFilename = os.path.join(args.save_directory,
                f"SegmentationPosToken_{config['Modality']}_{args.ModelName}_[DL{args.DINO_learning_rate:.3f}]_[CA{args.contrastive_alpha:.3f}]" +
                f"_{epoch+1:02d}_[{trn_loss:.3f}-{val_loss}].pth")
        torch.save(MODEL.state_dict(), saveFilename)
        lr_scheduler.step(val_loss)
    return

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Script with various default arguments.")   

    # Model configurations
    parser.add_argument('--ModelName', type=str, default="ProViCNet", help='ProViCNet, ProViNet, ProViCNet_LoRA, UCTransNet, UNet, SwinUNet, MISSFormer, TransUNet, NestedUNet, LeViTUnet')
    parser.add_argument('--vit_backbone', type=str, default='dinov2_s_reg', help='Specify the DINO ViT backbone, applied only for ProViDNet. Options include: [dinov2_s_reg, dinov2_b_reg, dinov2_l_reg, dinov2_g_reg].')
    parser.add_argument('--img_size', type=int, default=448, help=' Set the image size. For ProViDNet & ProViNet use 448, for UCTransNet, MISSFormer, etc., use 256.')
    parser.add_argument('--nClass', type=int, default=4, help='Number of classes')
    parser.add_argument('--nChannel', type=int, default=3, help='Number of channels')
    
    # Training configurations
    parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
    parser.add_argument('--Bag_batch_size', type=int, default=8, help='Indicates the number of images included in mixing slices for training.')
    parser.add_argument('--num_workers', type=int, default=8, help='Set the number of workers for DataLoader. Helps in parallel data loading')
    parser.add_argument('--small_batchsize', type=int, default=72, help='The actual slice-level batch size used in training as an input to the model')
    
    # Learning rate configurations
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the model training')
    parser.add_argument('--DINO_learning_rate', type=float, default=0.05, help="DINO backbone's learning rate (smaller than the main learning rate, default lr*0.01)")
    parser.add_argument('--train_class_weight', type=float, nargs=4, default=[0.02, 0.04, 0.50, 0.50], help='Training class weights (Background, Prostate, Gland)')
    parser.add_argument('--valid_class_weight', type=float, nargs=4, default=[0.00, 0.00, 0.50, 0.50], help='Validation class weights')
    parser.add_argument('--contrastive', type=bool, default=True, help='Use patch-level contrastive learning')
    parser.add_argument('--contrastive_alpha', type=float, default=0.05, help="Segmentation loss * (1-alpha) + contrastive loss * alpha")
    parser.add_argument('--max_pair', type=int, default=30, help='The number of patch pairs for Contrastive learing')
    
    parser.add_argument('--filter_background_prob', type=float, nargs=2, default=[0.20, 0.80], help='US/MRI: Set the probability threshold to filter out background slices (excluding gland and cancer areas) during training. 1.0 means include all background slices')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Pretrained weights')

    parser.add_argument('--US', type=bool, default=False, help='Use US data')
    parser.add_argument('--config', type=str, default="configs/config_train_T2.yaml", help='Config file for path of data')
    parser.add_argument('--save_directory', type=str, default='./ModelWeights/', help='Pretrained weights')
    parser.add_argument('--cuda_device', type=int, default=0, help='Specify CUDA visible devices')
    
    
    args = parser.parse_args()    
    args.filter_background_prob = {'TRUS':args.filter_background_prob[0], 'MRI':args.filter_background_prob[1]}
    args.filter_background_prob_valid = {'TRUS':1.0, 'MRI':1.0}
    if torch.cuda.is_available(): # Check GPU available
        print("Number of GPU(s) available:", torch.cuda.device_count())
        args.device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device else 'cuda:0')
    else:
        print("CUDA is not available.")
        args.device = torch.device('cpu')

    print(f"Arguments: {args}")
    config_file = args.config
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Configurations: {config}")

    main(args, config)
