import argparse
import json
import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from ProViCNet.ModelArchitectures.Models import GetModel
from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import FusionModalities, load_partial_weights
from util_functions.Prostate_DataGenerator import US_MRI_Generator, collate_prostate_position_CS, getData
from util_functions.utils_weighted import initialize_weights
from util_functions.train_functions import tensor_shuffle, getPatchTokens, OneBatchTraining_fusion

def main(args, configs):
    ###############################################################
    # 1. Data Loading
    ###############################################################
    # Load datasets for each modality (T2, ADC, DWI) using their config files
    TrainDatasets, ValidDatasets, Test_Datasets = dict(), dict(), dict()
    for Modality, config in configs.items():
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
        if config['Modality'] == 'DWI':
            TrainDataset.drop('101646_0001', inplace=True) # this data 
        TrainDatasets[Modality] = TrainDataset
        ValidDatasets[Modality] = ValidDataset

    # get intersection between all modalities
    for Dataset in [TrainDatasets, ValidDatasets, Test_Datasets]:
        common_patient = set(Dataset[next(iter(Dataset))].index)
        for Modality, Dataset_modal in Dataset.items():
            common_patient.intersection_update(set(Dataset_modal.index))
        common_patient = sorted(common_patient)
        for Modality, Dataset_modal in Dataset.items():
            Dataset[Modality] = Dataset_modal.loc[common_patient]
            Dataset[Modality].reset_index(drop=True, inplace=True)

    # Load Data Generators for three modalties to Dictionaries
    VALID_GENERATORs = dict()
    TRAIN_DATALOADERs, VALID_DATALOADERs = dict(), dict()
    
    for Modality, TrainDataset in TrainDatasets.items():
        TRAIN_GENERATOR = US_MRI_Generator(
                        imageFileName=TrainDataset['Image'],
                        glandFileName=TrainDataset['Gland'],
                        cancerFileName=TrainDataset['Cancer'],
                        modality=TrainDataset['Modality'],
                        cancerTo2=False, Augmentation=False,
                        filter_background_prob=args.filter_background_prob_valid, img_size=args.img_size)
        TRAIN_DATALOADER = DataLoader(TRAIN_GENERATOR, batch_size=args.Bag_batch_size,
                        shuffle=False , collate_fn=collate_prostate_position_CS, num_workers=args.num_workers)
        TRAIN_DATALOADERs[Modality] = TRAIN_DATALOADER
                        
    for Modality, ValidDataset in ValidDatasets.items():
        VALID_GENERATOR = US_MRI_Generator(
                        imageFileName=ValidDataset['Image'],
                        glandFileName=ValidDataset['Gland'],
                        cancerFileName=ValidDataset['Cancer'],
                        modality=ValidDataset['Modality'],
                        cancerTo2=False, Augmentation=False,
                        filter_background_prob=args.filter_background_prob_valid, img_size=args.img_size)
        VALID_GENERATORs[Modality] = VALID_GENERATOR
        VALID_DATALOADER = DataLoader(VALID_GENERATOR, batch_size=1,
                        shuffle=False, collate_fn=collate_prostate_position_CS, num_workers=args.num_workers)
        VALID_DATALOADERs[Modality] = VALID_DATALOADER

    ###############################################################
    # 2. Model Loading
    ###############################################################
    # Pretrained weights for each modality model
    ModelWeights = {
        'T2':  args.T2_weights,
        'ADC': args.ADC_weights,
        'DWI': args.DWI_weights
    }

    MODELs = dict()
    for Modality in ['T2', 'ADC', 'DWI']:
        MODEL = GetModel(args.ModelName, args.nClass, args.nChannel, args.img_size, vit_backbone=args.vit_backbone, contrastive=args.contrastive)
        MODEL = MODEL.to(args.device)
        ret = MODEL.load_state_dict(torch.load( ModelWeights[Modality] , map_location=args.device), strict=True)
        MODEL.eval()
        MODELs[Modality] = MODEL

    # Load fusion model and its pretrained weights
    MODEL_Fusion = FusionModalities(embedding_size=MODELs['T2'].embedding_size, target_channels=128, num_classes=args.nClass)
    initialize_weights(MODEL_Fusion)
    MODEL_Fusion = MODEL_Fusion.to(args.device)

    if args.pretrained_weights is not None:
        MODEL_Fusion.load_state_dict(torch.load(args.pretrained_weights, map_location=args.device), strict=True)
    # Load partial weights from individual modality models into corresponding fusion streams
    load_partial_weights(MODELs['T2'].head.segmentation_conv, MODEL_Fusion.Stream_T2W.UpsampleModule)
    load_partial_weights(MODELs['ADC'].head.segmentation_conv, MODEL_Fusion.Stream_ADC.UpsampleModule)
    load_partial_weights(MODELs['DWI'].head.segmentation_conv, MODEL_Fusion.Stream_DWI.UpsampleModule)

    # Set optimizer: use a lower learning rate for stream parameters
    params_streams = list(MODEL_Fusion.Stream_T2W.parameters()) + \
                     list(MODEL_Fusion.Stream_ADC.parameters()) + \
                     list(MODEL_Fusion.Stream_DWI.parameters())
    params_seg = list(MODEL_Fusion.FusionSegmentation.parameters())
    optimizer = torch.optim.Adam([
        {'params': params_streams, 'lr': args.learning_rate * args.Seg_lr},
        {'params': params_seg, 'lr': args.learning_rate}
    ])
    weights_trn = torch.tensor(args.train_class_weight, device=args.device)
    criterion_trn = DiceCELoss(to_onehot_y=True, softmax=True, weight=weights_trn)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    os.makedirs(args.save_directory, exist_ok=True)

    ###############################################################
    # 3. Fusion Training Loop
    ###############################################################
    
    for epoch in range(args.epoch):
        MODEL_Fusion.train()
        trn_losses, trn_image_count = 0.0, 0
        for (Image_T2, Label, Posit), (Image_ADC, Label_ADC, Posit_ADC), (Image_DWI, Label_DWI, Posit_DWI) in \
                tqdm(zip(TRAIN_DATALOADERs['T2'], TRAIN_DATALOADERs['ADC'], TRAIN_DATALOADERs['DWI']), total=len(TRAIN_DATALOADERs['T2'])):
            
            # Exception Handling for batch mismatch
            if Label.shape[0] != Image_ADC.shape[0] or Label.shape[0] != Image_DWI.shape[0]:
                assert AssertionError("Batch size mismatch")
            if not (torch.equal(Label, Label_ADC) or torch.equal(Label, Label_DWI)):
                raise AssertionError("Label_T2 does not match either Label_ADC or Label_DWI")
            if Posit.shape[0] != Posit_ADC.shape[0] or Posit.shape[0] != Posit_DWI.shape[0]:
                assert AssertionError("Batch size mismatch")
            if not (torch.equal(Posit, Posit_ADC) or torch.equal(Posit, Posit_DWI)):
                raise AssertionError("Posit_T2 does not match either Posit_ADC or Posit_DWI")
            
            Image_T2 , Label, Posit = tensor_shuffle(Image_T2 , Label, args.device, pos = Posit, shuffle=False)
            Image_ADC, Label, Posit = tensor_shuffle(Image_ADC, Label, args.device, pos = Posit, shuffle=False)
            Image_DWI, Label, Posit = tensor_shuffle(Image_DWI, Label, args.device, pos = Posit, shuffle=False)
            Tokens_T2  = getPatchTokens(MODELs['T2' ], Image_T2 , Posit, args).to(args.device)
            Tokens_ADC = getPatchTokens(MODELs['ADC'], Image_ADC, Posit, args).to(args.device)
            Tokens_DWI = getPatchTokens(MODELs['DWI'], Image_DWI, Posit, args).to(args.device)

            trn_loss = OneBatchTraining_fusion(Tokens_T2, Tokens_ADC, Tokens_DWI, Label, MODEL_Fusion, criterion_trn, optimizer)
            trn_losses += trn_loss
            trn_image_count += 1
    
        MODEL_Fusion.eval()
        val_loss, val_image_count = 0.0, 0
        for sample_idx in range(len(VALID_GENERATORs['T2'])):
            Image_T2 , Label    , Posit     = collate_prostate_position_CS([VALID_GENERATORs['T2' ].__getitem__(sample_idx)])
            Image_ADC, Label_ADC, Posit_ADC = collate_prostate_position_CS([VALID_GENERATORs['ADC'].__getitem__(sample_idx)])
            Image_DWI, Label_DWI, Posit_DWI = collate_prostate_position_CS([VALID_GENERATORs['DWI'].__getitem__(sample_idx)])

            if Label[:, 2:4].sum() == 0:
                continue
            Image_T2 , Label, Posit = tensor_shuffle(Image_T2 , Label, args.device, pos = Posit, shuffle=False)
            Image_ADC, Label, Posit = tensor_shuffle(Image_ADC, Label, args.device, pos = Posit, shuffle=False)
            Image_DWI, Label, Posit = tensor_shuffle(Image_DWI, Label, args.device, pos = Posit, shuffle=False)
            Tokens_T2  = getPatchTokens(MODELs['T2' ], Image_T2 , Posit, args).to(args.device)
            Tokens_ADC = getPatchTokens(MODELs['ADC'], Image_ADC, Posit, args).to(args.device)
            Tokens_DWI = getPatchTokens(MODELs['DWI'], Image_DWI, Posit, args).to(args.device)

            with torch.no_grad():
                # preds_T2  = MODELs['T2'] (Image_T2 , Posit).cpu()
                # preds_ADC = MODELs['ADC'](Image_ADC, Posit).cpu()
                # preds_DWI = MODELs['DWI'](Image_DWI, Posit).cpu()
                preds_MP  = MODEL_Fusion(Tokens_T2, Tokens_ADC, Tokens_DWI).cpu()
        
            Label_argmax = Label.argmax(axis=1).unsqueeze(1)
            val_loss += criterion_trn(preds_MP, Label_argmax).item()
            val_image_count += 1
        val_loss /= val_image_count

        # Evaluate fusion performance on the validation set using ROC-AUC and Dice metrics
        save_fname = os.path.join(args.save_directory,
            f"Fusion_{configs['T2']['Modality']}_{args.ModelName}_ep{epoch+1:02d}_loss[{trn_loss:.3f}_{val_loss:.3f}].pth")
        torch.save(MODEL_Fusion.state_dict(), save_fname)
        lr_scheduler.step(val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion Training Script for ADC, DWI, and T2 models.")
    # Model configurations
    parser.add_argument('--ModelName', type=str, default="ProViCNet", help="Model architecture (e.g., ProViDNet, UCTransNet).")
    parser.add_argument('--vit_backbone', type=str, default='dinov2_s_reg', help="DINO ViT backbone (options: dinov2_s_reg, dinov2_b_reg, dinov2_l_reg, dinov2_g_reg).")
    parser.add_argument('--img_size', type=int, default=448, help="Image size (448 for ProViDNet/ProViCNet; 256 for others).")
    parser.add_argument('--nClass', type=int, default=4, help="Number of segmentation classes.")
    parser.add_argument('--nChannel', type=int, default=3, help="Number of input channels.")
    
    # Training configurations
    parser.add_argument('--Bag_batch_size', type=int, default=4, help="Batch size for mixed-slice training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument('--small_batchsize', type=int, default=36, help="Slice-level batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Base learning rate.")
    parser.add_argument('--Seg_lr', type=float, default=0.25, help="Learning rate scaling factor for fusion streams.")
    parser.add_argument('--epoch', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--train_class_weight', type=float, nargs=4, default=[0.02, 0.04, 0.50, 0.50], help="Training class weights (e.g., [Background, Prostate, Gland, ...]).")
    parser.add_argument('--valid_class_weight', type=float, nargs=4, default=[0.00, 0.00, 0.50, 0.50], help="Validation class weights.")
    parser.add_argument('--filter_background_prob', type=float, nargs=2, default=[0.20, 1.00], help="Background filtering thresholds (first for TRUS, second for MRI).")
    parser.add_argument('--save_directory', type=str, default='./MODEL_CS_Contrastive_Fusion/', help="Directory to save fusion model checkpoints.")
    parser.add_argument('--cuda_device', type=int, default=4, help="CUDA device index to use.")
    parser.add_argument('--config_files', type=str, nargs='+', default=["configs/config_train_T2.yaml", "configs/config_train_ADC.yaml", "configs/config_train_DWI.yaml"],
                            help="Paths to config files with data paths.")
    parser.add_argument('--contrastive', type=int, default=1, help="Enable contrastive learning (1 to enable).")

    # model weights
    parser.add_argument('--pretrained_weights', type=str, default=None, help="Path to pretrained weights (if available).")
    parser.add_argument('--T2_weights', type=str, default=None, help="Path to pretrained T2 weights.")
    parser.add_argument('--ADC_weights', type=str, default=None, help="Path to pretrained ADC weights.")
    parser.add_argument('--DWI_weights', type=str, default=None, help="Path to pretrained DWI weights.")
    
    args = parser.parse_args()
    args.filter_background_prob = {'TRUS': args.filter_background_prob[0],
                                   'MRI': args.filter_background_prob[1]}
    args.filter_background_prob_valid = {'TRUS': 1.0, 'MRI': 1.0}
    if torch.cuda.is_available():
        print("Number of GPU(s) available:", torch.cuda.device_count())
        args.device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device else 'cuda:0')
    else:
        print("CUDA is not available.")
        args.device = torch.device('cpu')

    # Load individual config files for T2, ADC, and DWI
    configs = {}
    for cfg_file in args.config_files:
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
        configs[cfg['Modality']] = cfg
        print(f"Loaded config for {cfg['Modality']}")
        
    main(args, configs)
