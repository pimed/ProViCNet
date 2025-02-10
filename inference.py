import argparse
import os
import numpy as np
from ProViCNet.ModelArchitectures.Models import GetModel
from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import FusionModalities, load_partial_weights
from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator

from ProViCNet.util_functions.utils_weighted import initialize_weights
from ProViCNet.util_functions.inference import ProViCNet_Inference, ProViCNet_data_preparation, visualize_max_cancer, saveData

import torch
from tqdm import tqdm
import random

def main(args):
    # 1. Data Loading #############################################
    # Load the dataset, with names manully specified
    Dataset = dict({
        'TRUS': [ 'dataset_TRUS/100648_0001_trus.nii.gz' ],
        'Gland': ['dataset_TRUS/100648_0001_trus_prostate_label.nii.gz'],
        'Cancer': ['dataset_TRUS/100648_0001_trus_roi_bxconfirmed_label.nii.gz'], ## optional for visualization
    })

    TEST_GENERATORs = dict()
    for Modal in ['TRUS']:
        TEST_GENERATORs[Modal] = US_MRI_Generator(
                        imageFileName=Dataset[Modal],
                        glandFileName=Dataset['Gland'],
                        cancerFileName=Dataset['Cancer'], # optional for visualization,
                        modality='TRUS',
                        cancerTo2=False, Augmentation=False,
                        img_size=args.img_size, nChannel=args.nChannel)
        
    ###############################################################
    # 2. Model Loading ############################################
    print("# 2. DINO Segmentation Model Loading ############## ") #
    ###############################################################
    # Load individual models
    ModelWeights = dict({
        'TRUS': os.path.join(args.model_path, 'TRUS_best.pth'),
    })

    MODEL = GetModel(args.ModelName, args.nClass, args.nChannel, args.img_size, vit_backbone=args.vit_backbone, contrastive=args.contrastive, US=True)
    MODEL = MODEL.to(args.device)
    ret = MODEL.load_state_dict(torch.load( ModelWeights[Modal] , map_location=args.device), strict=True)
    print('Model load:', ret)
    MODEL.eval()

    # Load fusion model
    for sample_idx in tqdm(range(len(TEST_GENERATORs['TRUS']))):
        # Data load T2, ADC, DWI with 3-consecutive slices, and Segmentation Labels, and axial-position information
        Image_T2, Image_ADC, Image_DWI, Posit, Label = ProViCNet_data_preparation(sample_idx, args, TEST_GENERATORs, modality='TRUS')
        preds_T2_softmax, preds_ADC_softmax, preds_DWI_softmax, preds_MP_softmax = \
            ProViCNet_Inference(Image_T2, Image_ADC, Image_DWI, Posit, args, MODELs, MODEL_Fusion)

        os.makedirs(args.save_folder, exist_ok=True)
        patient_name = os.path.join(
            args.save_folder,
            os.path.basename(TEST_GENERATORs['T2'].imageFileName[sample_idx]).split('_t2')[0]
        )

        # Save Probability Maps
        filename_T2    = patient_name + '_ProViCNet_T2_Probability.nii.gz'
        filename_ADC   = patient_name + '_ProViCNet_ADC_Probability.nii.gz'
        filename_DWI   = patient_name + '_ProViCNet_DWI_Probability.nii.gz'
        filename_mpMRI = patient_name + '_ProViCNet_mpMRI_Probability.nii.gz'
        
        saveData(preds_T2_softmax[:,2], TEST_GENERATORs['T2'].imageFileName[sample_idx], filename_T2)
        saveData(preds_ADC_softmax[:,2], TEST_GENERATORs['ADC'].imageFileName[sample_idx], filename_ADC)
        saveData(preds_DWI_softmax[:,2], TEST_GENERATORs['DWI'].imageFileName[sample_idx], filename_DWI)
        saveData(preds_MP_softmax[:,2], TEST_GENERATORs['T2'].imageFileName[sample_idx], filename_mpMRI)

        # Save Predicted Label Maps

        filename_T2    = patient_name + '_ProViCNet_T2_PredLabel.nii.gz'
        filename_ADC   = patient_name + '_ProViCNet_ADC_PredLabel.nii.gz'
        filename_DWI   = patient_name + '_ProViCNet_DWI_PredLabel.nii.gz'
        filename_mpMRI = patient_name + '_ProViCNet_mpMRI_PredLabel.nii.gz'
        
        saveData((preds_T2_softmax [:,2] > args.threshold).astype(float), TEST_GENERATORs['T2' ].imageFileName[sample_idx], filename_T2)
        saveData((preds_ADC_softmax[:,2] > args.threshold).astype(float), TEST_GENERATORs['ADC'].imageFileName[sample_idx], filename_ADC)
        saveData((preds_DWI_softmax[:,2] > args.threshold).astype(float), TEST_GENERATORs['DWI'].imageFileName[sample_idx], filename_DWI)
        saveData((preds_MP_softmax [:,2] > args.threshold).astype(float), TEST_GENERATORs['T2' ].imageFileName[sample_idx], filename_mpMRI)

        # Save Visualization
        filename_visualization = patient_name + '_Visualization.png'
        visualize_max_cancer(
            Image_T2, Image_ADC, Image_DWI, Label,
            preds_T2_softmax, preds_ADC_softmax, preds_DWI_softmax, preds_MP_softmax,
            filename_visualization
        )

    return

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Script with various default arguments.")   

    # Model configurations
    parser.add_argument('--ModelName', type=str, default="ProViDNet", help='Choose the segmentation model architecture. Examples include ProViDNet, UCTransNet, etc. For more options, refer to `ModelArchitectures/Models.py`')
    parser.add_argument('--vit_backbone', type=str, default='dinov2_s_reg', help='Specify the DINO ViT backbone, applied only for ProViDNet. Options include: [dinov2_s_reg, dinov2_b_reg, dinov2_l_reg, dinov2_g_reg].')
    parser.add_argument('--img_size', type=int, default=448, help=' Set the image size. For ProViDNet & ProViCNet use 448, for UCTransNet, MISSFormer, etc., use 256.')
    parser.add_argument('--nClass', type=int, default=4, help='Number of classes (Background, Prostate gland, Cancer)')
    parser.add_argument('--nChannel', type=int, default=9, help='Number of channels (number of consecutive slices), default: 3')
    parser.add_argument('--contrastive', type=int, default=1, help='Contrastive learning')
    
    # Training configurations
    parser.add_argument('--cuda_device', type=int, default=0, help='Specify CUDA visible devices')
    parser.add_argument('--small_batchsize', type=int, default=32, help='Number of epochs')
    
    # Inference configurations
    parser.add_argument('--save_pred_visualization', type=int, default=0, help='Save visualization')
    parser.add_argument('--save_folder', type=str, default='results_ProViCNet/', help='Save folder')
    parser.add_argument('--threshold', type=float, default=0.4, help='cut-off threshold for classification')
    parser.add_argument('--model_path', type=str, default='./ModelWeights/', help='Path to the model weights')

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    set_seed(42)
    main(args)

# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality T2 --cuda_device 0 --addname 1 --pretrained_weights /home/sosal/student_projects/JeongHoonLee/ProViDNet/MODEL_CS_Contrastive_All/All/SegmentationPosToken_T2_ProViDNet_All_[DL0.010]_[CA0.050]_02_[0.295]_[0.891_0.353]_[0.926_0.397].pth &

# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality T2 --cuda_device 0 --addname 1 --pretrained_weights /home/sosal/student_projects/JeongHoonLee/ProViDNet/MODEL_CS_Contrastive_All/All/SegmentationPosToken_T2_ProViDNet_All_[DL0.010]_[CA0.050]_02_[0.295]_[0.891_0.353]_[0.926_0.397].pth &


# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 0 --addname 1 --pretrained_weights AllDatasetPosToken_Fusion_02_[640.370]_[0.947_0.472]_[0.950_0.458].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 0 --addname 2 --pretrained_weights AllDatasetPosToken_Fusion_03_[846.908]_[0.947_0.473]_[0.951_0.464].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 1 --addname 3 --pretrained_weights AllDatasetPosToken_Fusion_04_[1047.677]_[0.942_0.475]_[0.951_0.467].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 1 --addname 4 --pretrained_weights AllDatasetPosToken_Fusion_13_[4324.785]_[0.946_0.471]_[0.950_0.463].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 2 --addname 5 --pretrained_weights AllDatasetPosToken_Fusion_14_[4530.250]_[0.945_0.471]_[0.950_0.465].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 2 --addname 6 --pretrained_weights AllDatasetPosToken_Fusion_15_[4733.566]_[0.945_0.472]_[0.950_0.467].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 3 --addname 7 --pretrained_weights AllDatasetPosToken_Fusion_16_[4935.188]_[0.945_0.472]_[0.951_0.468].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 3 --addname 8 --pretrained_weights AllDatasetPosToken_Fusion_11_[3904.797]_[0.951_0.469]_[0.944_0.460].pth &
# nohup python infer_all.py --save_folder Inference_results/selectBest/All/ --modality mpMRI --cuda_device 4 --addname 9 --pretrained_weights AllDatasetPosToken_Fusion_20_[5730.684]_[0.940_0.473]_[0.951_0.470].pth &
