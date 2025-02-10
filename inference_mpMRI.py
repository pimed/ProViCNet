import argparse
import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Import model, data generator, and utility functions from ProViCNet
from ProViCNet.ModelArchitectures.Models import GetModel
from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import FusionModalities, load_partial_weights
from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator
from ProViCNet.util_functions.utils_weighted import set_seed
from ProViCNet.util_functions.inference import (
    ProViCNet_Inference,
    ProViCNet_data_preparation,
    visualize_max_cancer,
    saveData,
    visualize_featuremap
)
from ProViCNet.util_functions.train_functions import getPatchTokens

def load_weight_from_url(url, device):
    """
    Downloads the weight file from Hugging Face Hub and loads it.
    Assumes URL format: "https://huggingface.co/{repo_id}/resolve/main/{filename}"
    """
    parts = url.split('/')
    repo_id = f"{parts[3]}/{parts[4]}"
    filename = parts[-1]
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return torch.load(weight_path, map_location=device)

def main(args):
    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    # Load dataset paths from configuration file.
    Dataset = args.config['sample_dataset']
    TEST_GENERATORs = dict()
    for Sequence in ['T2', 'ADC', 'DWI']:
        TEST_GENERATORs[Sequence] = US_MRI_Generator(
            imageFileName=Dataset[Sequence],
            glandFileName=Dataset['Gland'],
            cancerFileName=Dataset['Cancer'],  # Optional for visualization.
            modality='MRI',
            cancerTo2=False,
            Augmentation=False,
            img_size=args.img_size
        )
    
    # -------------------------------------------------------------------------
    # 2. Model Loading
    # -------------------------------------------------------------------------
    # Load individual modality models (T2, ADC, DWI)
    MODELs = dict()
    for Sequence in ['T2', 'ADC', 'DWI']:
        model = GetModel(
            args.ModelName,
            args.nClass,
            args.nChannel,
            args.img_size,
            vit_backbone=args.vit_backbone,
            contrastive=args.contrastive
        )
        model = model.to(args.device)
        state_dict = load_weight_from_url(args.config['model_weights'][Sequence], args.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        MODELs[Sequence] = model

    # Load fusion model for MRI sequences (mpMRI)
    MODEL_Fusion = FusionModalities(
        embedding_size=MODELs['T2'].embedding_size,
        target_channels=128,
        num_classes=args.nClass
    )
    state_dict = load_weight_from_url(args.config['model_weights']['mpMRI'], args.device)
    MODEL_Fusion.load_state_dict(state_dict, strict=True)
    MODEL_Fusion = MODEL_Fusion.to(args.device)
    load_partial_weights(MODELs['T2'].head.segmentation_conv, MODEL_Fusion.Stream_T2W.UpsampleModule)
    load_partial_weights(MODELs['ADC'].head.segmentation_conv, MODEL_Fusion.Stream_ADC.UpsampleModule)
    load_partial_weights(MODELs['DWI'].head.segmentation_conv, MODEL_Fusion.Stream_DWI.UpsampleModule)
    MODEL_Fusion.eval()

    # -------------------------------------------------------------------------
    # 3. Model Inference
    # -------------------------------------------------------------------------
    # 3.1. Patch-level Feature Extraction and Visualization
    for sample_idx in range(len(TEST_GENERATORs['T2'])):
        Image_T2, Image_ADC, Image_DWI, Posit, Label = ProViCNet_data_preparation(
            sample_idx, args, TEST_GENERATORs, modality='MRI')
        Tokens_T2  = getPatchTokens(MODELs['T2'], Image_T2, Posit, args).to(args.device)
        Tokens_ADC = getPatchTokens(MODELs['ADC'], Image_ADC, Posit, args).to(args.device)
        Tokens_DWI = getPatchTokens(MODELs['DWI'], Image_DWI, Posit, args).to(args.device)
        
        patient_ID = os.path.basename(TEST_GENERATORs['T2'].imageFileName[sample_idx]).split('_t2')[0]
        visualize_featuremap(Tokens_T2, Image_T2, Label,
            os.path.join(args.visualization_folder, f'{patient_ID}_featuremap_T2.png'))
        visualize_featuremap(Tokens_ADC, Image_ADC, Label,
            os.path.join(args.visualization_folder, f'{patient_ID}_featuremap_ADC.png'))
        visualize_featuremap(Tokens_DWI, Image_DWI, Label,
            os.path.join(args.visualization_folder, f'{patient_ID}_featuremap_DWI.png'))
    
    # 3.2. Full Inference: Cancer Segmentation and Visualization
    os.makedirs(args.save_folder, exist_ok=True)
    for sample_idx in range(len(TEST_GENERATORs['T2'])):
        Image_T2, Image_ADC, Image_DWI, Posit, Label = ProViCNet_data_preparation(
            sample_idx, args, TEST_GENERATORs)
        
        preds_T2_softmax, preds_ADC_softmax, preds_DWI_softmax, preds_MP_softmax = ProViCNet_Inference(
            Image_T2, Image_ADC, Image_DWI, Posit, args, MODELs, MODEL_Fusion, args.only_csPCa)
        patient_ID = os.path.basename(TEST_GENERATORs['T2'].imageFileName[sample_idx]).split('_t2')[0]
        
        # Save predicted probability maps (using channel 2).
        prob_filename_T2    = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_T2_probability.nii.gz')
        prob_filename_ADC   = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_ADC_probability.nii.gz')
        prob_filename_DWI   = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_DWI_probability.nii.gz')
        prob_filename_mpMRI = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_mpMRI_probability.nii.gz')
        
        saveData(preds_T2_softmax[:, 2], TEST_GENERATORs['T2'].imageFileName[sample_idx], prob_filename_T2)
        saveData(preds_ADC_softmax[:, 2], TEST_GENERATORs['ADC'].imageFileName[sample_idx], prob_filename_ADC)
        saveData(preds_DWI_softmax[:, 2], TEST_GENERATORs['DWI'].imageFileName[sample_idx], prob_filename_DWI)
        saveData(preds_MP_softmax[:, 2], TEST_GENERATORs['T2'].imageFileName[sample_idx], prob_filename_mpMRI)
        
        # Save predicted label maps (using thresholding on channel 2).
        label_filename_T2    = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_T2_predLabel.nii.gz')
        label_filename_ADC   = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_ADC_predLabel.nii.gz')
        label_filename_DWI   = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_DWI_predLabel.nii.gz')
        label_filename_mpMRI = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_mpMRI_predLabel.nii.gz')
        
        saveData((preds_T2_softmax[:, 2] > args.threshold).float(),
                 TEST_GENERATORs['T2'].imageFileName[sample_idx],label_filename_T2)
        saveData((preds_ADC_softmax[:, 2] > args.threshold).float(),
                 TEST_GENERATORs['ADC'].imageFileName[sample_idx],label_filename_ADC)
        saveData((preds_DWI_softmax[:, 2] > args.threshold).float(),
                 TEST_GENERATORs['DWI'].imageFileName[sample_idx],label_filename_DWI)
        saveData((preds_MP_softmax[:, 2] > args.threshold).float(),
                 TEST_GENERATORs['T2'].imageFileName[sample_idx],label_filename_mpMRI)
        
        # Save visualization of segmentation results.
        vis_filename = os.path.join(args.visualization_folder, f'{patient_ID}_mpMRI_visualization.png')
        visualize_max_cancer(
            Image_T2, Image_ADC, Image_DWI, Label,
            preds_T2_softmax, preds_ADC_softmax, preds_DWI_softmax, preds_MP_softmax,
            vis_filename
        )
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProViCNet MRI Inference Script")
    
    # Model configurations
    parser.add_argument('--ModelName', type=str, default="ProViCNet",
        help="Segmentation model architecture (e.g., ProViCNet, UCTransNet, etc.).")
    parser.add_argument('--vit_backbone', type=str, default='dinov2_s_reg',
        help="DINO ViT backbone for ProViCNet (e.g., dinov2_s_reg, dinov2_b_reg, dinov2_l_reg, dinov2_g_reg).")
    parser.add_argument('--img_size', type=int, default=448, help="Image size; use 448 for ProViCNet, 256 for others.")
    parser.add_argument('--nClass', type=int, default=4, help="Number of segmentation classes.")
    parser.add_argument('--nChannel', type=int, default=3, help="Number of channels (consecutive slices).")
    parser.add_argument('--contrastive', type=bool, default=True, help="Use patch-level contrastive learning.")

    # Training configurations (if applicable)
    parser.add_argument('--cuda_device', type=int, default=0,help="CUDA device index to use.")
    parser.add_argument('--only_csPCa', type=bool, default=False, help="Keep only csPCa or merge all cancer channels.")

    # Inference and output configurations
    parser.add_argument('--save_folder', type=str, default='results_ProViCNet/', help="Folder to save predicted outputs.")
    parser.add_argument('--visualization_folder', type=str, default='visualization_ProViCNet/', help="Folder to save visualization images.")
    parser.add_argument('--threshold', type=float, default=0.4, help="Threshold for segmentation.")
    parser.add_argument('--small_batchsize', type=int, default=16, help="Batch size for inference.")
    parser.add_argument('--config_file', type=str, default='configs/config_infer_MRI.yaml', help="Path to YAML configuration file with dataset and weight URLs.")

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    with open(args.config_file) as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)
    
    set_seed(42)
    main(args)
