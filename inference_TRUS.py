import argparse
import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Import model, data generator, and utility functions from ProViCNet
from ProViCNet.ModelArchitectures.Models import GetModel
from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator
from ProViCNet.util_functions.utils_weighted import set_seed
from ProViCNet.util_functions.inference import (
    ProViCNet_data_preparation,
    visualize_TRUS,
    saveData,
    visualize_featuremap,
    merge_cancer,
    keep_csPCa_only
)
from ProViCNet.util_functions.train_functions import getPatchTokens_TRUS

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
    
    # Currently only TRUS modality is used.
    for modal in ['TRUS']:
        TEST_GENERATORs[modal] = US_MRI_Generator(
            imageFileName=Dataset[modal],
            glandFileName=Dataset['Gland'],
            cancerFileName=Dataset['Cancer'],  # Optional for visualization.
            modality=modal,
            cancerTo2=False,
            Augmentation=False,
            img_size=args.img_size,
            nChannel=args.nChannel
        )
    
    # -------------------------------------------------------------------------
    # 2. Model Loading
    # -------------------------------------------------------------------------
    # Load the TRUS segmentation model.
    MODEL = GetModel(
        args.ModelName,
        args.nClass,
        args.nChannel,
        args.img_size,
        vit_backbone=args.vit_backbone,
        contrastive=args.contrastive,
        US=True
    )
    MODEL = MODEL.to(args.device)
    
    # Use the weight URL corresponding to the current modality ('TRUS').
    state_dict = load_weight_from_url(args.config['model_weights'][modal], args.device)
    MODEL.load_state_dict(state_dict, strict=True)
    MODEL.eval()
    
    # -------------------------------------------------------------------------
    # 3. Model Inference: Feature Extraction and Prediction
    # -------------------------------------------------------------------------
    # 3.1. Patch-level Feature Extraction and Visualization
    for sample_idx in range(len(TEST_GENERATORs['TRUS'])):
        Image_TRUS, Label, Posit = ProViCNet_data_preparation(
            sample_idx, args, TEST_GENERATORs, modality='TRUS')
        
        # Extract patch tokens from TRUS image.
        Tokens_TRUS = getPatchTokens_TRUS(MODEL, Image_TRUS, Posit, args).to(args.device)
        
        # Derive patient ID from file name.
        patient_ID = os.path.basename(TEST_GENERATORs['TRUS'].imageFileName[sample_idx]).split('_trus')[0]
        
        # Visualize the extracted feature map using UMAP.
        featuremap_filename = os.path.join(args.visualization_folder, f'{patient_ID}_featuremap_TRUS.png')
        visualize_featuremap(Tokens_TRUS, Image_TRUS, Label, featuremap_filename)
    
    # 3.2. Full Inference, Prediction, and Visualization
    os.makedirs(args.save_folder, exist_ok=True)
    for sample_idx in range(len(TEST_GENERATORs['TRUS'])):
        Image_TRUS, Label, Posit = ProViCNet_data_preparation(
            sample_idx, args, TEST_GENERATORs, modality='TRUS')
        
        with torch.no_grad():
            # Obtain model predictions.
            pred_TRUS = MODEL(Image_TRUS, pos=Posit).cpu()
            preds_TRUS_softmax = torch.softmax(pred_TRUS, dim=1)
            # Process predictions based on whether only csPCa should be kept.
            if args.only_csPCa:
                preds_TRUS_softmax = keep_csPCa_only(preds_TRUS_softmax)
            else:
                preds_TRUS_softmax = merge_cancer(preds_TRUS_softmax)
        
        patient_ID = os.path.basename(TEST_GENERATORs['TRUS'].imageFileName[sample_idx]).split('_trus')[0]
        
        # Save predicted probability map (using channel 2).
        prob_filename = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_TRUS_probability.nii.gz')
        saveData(preds_TRUS_softmax[:, 2], TEST_GENERATORs['TRUS'].imageFileName[sample_idx], prob_filename)
        
        # Save predicted label map (using thresholding on channel 2).
        label_filename = os.path.join(args.save_folder, f'{patient_ID}_ProViCNet_TRUS_predLabel.nii.gz')
        saveData((preds_TRUS_softmax[:, 2] > args.threshold).float(),
                 TEST_GENERATORs['TRUS'].imageFileName[sample_idx], label_filename)
        
        # Visualize predictions with overlay on the original image.
        vis_filename = os.path.join(args.visualization_folder, f'{patient_ID}_TRUS_visualization.png')
        visualize_TRUS(Image_TRUS, Label, preds_TRUS_softmax, vis_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRUS Inference Script for ProViCNet")
    
    # Model configurations
    parser.add_argument('--ModelName', type=str, default="ProViCNet",
        help="Segmentation model architecture (e.g., ProViDNet, UCTransNet, etc.).")
    parser.add_argument('--vit_backbone', type=str, default='dinov2_s_reg',
        help="DINO ViT backbone for ProViDNet (e.g., dinov2_s_reg, dinov2_b_reg, dinov2_l_reg, dinov2_g_reg).")
    parser.add_argument('--img_size', type=int, default=448, help="Image size; use 448 for ProViDNet/ProViCNet, or 256 for others.")
    parser.add_argument('--nClass', type=int, default=4, help="Number of segmentation classes (e.g., Background, Prostate gland, Cancer).")
    parser.add_argument('--nChannel', type=int, default=9, help="Number of channels (consecutive slices); default is 9.")
    parser.add_argument('--contrastive', type=int, default=1, help="Enable or disable contrastive learning (1 or 0).")
    
    # Training configurations (if applicable)
    parser.add_argument('--cuda_device', type=int, default=0, help="CUDA device index to use.")
    parser.add_argument('--only_csPCa', type=bool, default=False,
        help="If True, keep only csPCa; otherwise merge cancer channels.")
    
    # Inference and output configurations
    parser.add_argument('--save_folder', type=str, default='results_ProViCNet/', help="Folder to save predicted outputs.")
    parser.add_argument('--visualization_folder', type=str, default='visualization_ProViCNet/', help="Folder to save visualization images.")
    parser.add_argument('--threshold', type=float, default=0.4, help="Threshold for converting probabilities to binary labels.")
    parser.add_argument('--small_batchsize', type=int, default=16, help="Batch size for inference.")
    parser.add_argument('--config_file', type=str, default='configs/config_infer_TRUS.yaml', help="Path to YAML configuration file with dataset and weight URLs.")
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    # Load configuration file
    with open(args.config_file) as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)
    
    set_seed(42)
    main(args)
