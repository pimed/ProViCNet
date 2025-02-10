from ProViCNet.util_functions.train_functions import tensor_shuffle, getPatchTokens
from ProViCNet.util_functions.Prostate_DataGenerator import collate_prostate_position_CS

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch
import cv2
import SimpleITK as sitk
import umap

def check_conditions(Image_T2, Image_ADC, Image_DWI, Label_T2, Label_ADC, Label_DWI, Posit_T2, Posit_ADC, Posit_DWI):
    size_match = (
        Label_T2.shape[0] == Image_ADC.shape[0] == Image_DWI.shape[0] and
        Posit_T2.shape[0] == Posit_ADC.shape[0] == Posit_DWI.shape[0]
    )
    if size_match == False:
        print("Size mismatch")
        return True

    labels_match = torch.equal(Label_T2, Label_ADC) or torch.equal(Label_T2, Label_DWI)
    if labels_match == False:
        print("Labels mismatch")
        return True

    has_nan = torch.isnan(Image_T2).any() or torch.isnan(Image_ADC).any() or torch.isnan(Image_DWI).any()
    if has_nan:
        print("Has NaN")
        return True
    
    T2_maxSlice = Image_T2[Label_T2[:,1].sum(axis=(1,2)).argmax(), 1]
    ADC_maxSlice = Image_ADC[Label_T2[:,1].sum(axis=(1,2)).argmax(), 1]
    DWI_maxSlice = Image_DWI[Label_T2[:,1].sum(axis=(1,2)).argmax(), 1]
    noInfo = T2_maxSlice.max() - T2_maxSlice.min() < 0.2 or ADC_maxSlice.max() - ADC_maxSlice.min() < 0.2 or DWI_maxSlice.max() - DWI_maxSlice.min() < 0.2
    if noInfo:
        print("No information")
        return True
    return not (size_match and labels_match) or has_nan or noInfo


def ProViCNet_data_preparation(sample_idx, args, TEST_GENERATORs, modality='MRI'):
    if modality == 'MRI':
        T2_batch  = TEST_GENERATORs['T2'].__getitem__(sample_idx)
        ADC_batch = TEST_GENERATORs['ADC'].__getitem__(sample_idx)
        DWI_batch = TEST_GENERATORs['DWI'].__getitem__(sample_idx)
        Image_T2 , Label_T2 , Posit_T2  = collate_prostate_position_CS([T2_batch])
        Image_ADC, Label_ADC, Posit_ADC = collate_prostate_position_CS([ADC_batch])
        Image_DWI, Label_DWI, Posit_DWI = collate_prostate_position_CS([DWI_batch])

        if check_conditions(Image_T2, Image_ADC, Image_DWI, Label_T2, Label_ADC, Label_DWI, Posit_T2, Posit_ADC, Posit_DWI):
            print("##################################")
            print("## Error in dataset # ProViCNet ##")
            print("##################################")
            return False

        # Change data type tensor float. No shuffle
        Label, Posit = Label_T2, Posit_T2
        Image_T2 , Label, Posit = tensor_shuffle(Image_T2 , Label, args.device, pos = Posit, shuffle=False)
        Image_ADC, Label, Posit = tensor_shuffle(Image_ADC, Label, args.device, pos = Posit, shuffle=False)
        Image_DWI, Label, Posit = tensor_shuffle(Image_DWI, Label, args.device, pos = Posit, shuffle=False)
        return Image_T2, Image_ADC, Image_DWI, Posit, Label
    elif modality == 'TRUS':
        TRUS_batch  = TEST_GENERATORs['TRUS'].__getitem__(sample_idx)
        Image_TRUS , Label_TRUS, Posit_TRUS  = collate_prostate_position_CS([TRUS_batch])
        Image_TRUS , Label_TRUS, Posit_TRUS = tensor_shuffle(Image_TRUS , Label_TRUS, args.device, pos = Posit_TRUS, shuffle=False)
        return Image_TRUS, Label_TRUS, Posit_TRUS

def keep_csPCa_only(pred):
    return torch.cat((pred[:, :2], pred[:, 3:4]), dim=1)

def merge_cancer(pred):
    return torch.cat((pred[:, :2], (pred[:, 2] + pred[:, 3]).unsqueeze(1)), dim=1)


def ProViCNet_Inference(Image_T2, Image_ADC, Image_DWI, Posit, args, MODELs, MODEL_Fusion, only_csPCa=False):
    
    # get Patch tokens and predictions from each MRI Sequences.
    # MP uses T2, ADC, DWI patch tokens together to predict the 'mpMRI' segmentation map.
    with torch.no_grad():
        Tokens_T2  = getPatchTokens(MODELs['T2' ], Image_T2 , Posit, args).to(args.device)
        Tokens_ADC = getPatchTokens(MODELs['ADC'], Image_ADC, Posit, args).to(args.device)
        Tokens_DWI = getPatchTokens(MODELs['DWI'], Image_DWI, Posit, args).to(args.device)
        
        preds_T2  = MODELs['T2'] (Image_T2 , Posit).cpu() ## Output from T2 model
        preds_ADC = MODELs['ADC'](Image_ADC, Posit).cpu() ## Output from ADC model
        preds_DWI = MODELs['DWI'](Image_DWI, Posit).cpu() ## Output from DWI model
        preds_MP  = MODEL_Fusion(Tokens_T2, Tokens_ADC, Tokens_DWI).cpu() ## Output from mpMRI model

    preds_T2_softmax = torch.softmax(preds_T2, dim=1)
    preds_ADC_softmax = torch.softmax(preds_ADC, dim=1)
    preds_DWI_softmax = torch.softmax(preds_DWI, dim=1)        
    preds_MP_softmax = torch.softmax(preds_MP, dim=1)

    if only_csPCa:
        preds_T2_softmax = keep_csPCa_only(preds_T2_softmax)
        preds_ADC_softmax = keep_csPCa_only(preds_ADC_softmax)
        preds_DWI_softmax = keep_csPCa_only(preds_DWI_softmax)
        preds_MP_softmax = keep_csPCa_only(preds_MP_softmax)
    else:
        preds_T2_softmax = merge_cancer(preds_T2_softmax)
        preds_ADC_softmax = merge_cancer(preds_ADC_softmax)
        preds_DWI_softmax = merge_cancer(preds_DWI_softmax)
        preds_MP_softmax = merge_cancer(preds_MP_softmax)
    
        

    return preds_T2_softmax, preds_ADC_softmax, preds_DWI_softmax, preds_MP_softmax


def visualize_max_cancer(Image_T2, Image_ADC, Image_DWI, Label,
              preds_T2_softmax, preds_ADC_softmax, preds_DWI_softmax, preds_MP_softmax, filename):
    # assume that prediction is 4-channel
    jet = plt.cm.jet
    colors = jet(np.arange(256))
    colors[:int(0.15*256), -1] = 0
    jet_transp = LinearSegmentedColormap.from_list("custom_jet", colors)

    colors = [(0,0,0,0), (1,1,0,1), (1,0,0,1)]  # transparent, and yellow, red
    label_cmap = ListedColormap(colors)
    
    slice_idx = Label[:, 2:4].sum(axis=(1,2,3)).argmax() # select maximum slice index
    segmentation_map = Label.argmax(axis=1).cpu().numpy()[slice_idx]
    segmentation_map[segmentation_map > 2] = 2
    mask_combined = np.logical_or(segmentation_map == 1, segmentation_map == 2).astype(np.uint8)
    mask_label2 = (segmentation_map == 2).astype(np.uint8)

    contours_combined, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_label2, _ = cv2.findContours(mask_label2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = np.zeros(segmentation_map.shape, dtype=np.uint8)
    cv2.drawContours(contour_image, contours_combined, -1, 1, 1)
    cv2.drawContours(contour_image, contours_label2, -1, 2, 1)

    fig, axes = plt.subplots(2, 4, figsize=(15,8))
    for i in range(2):
        axes[i,0].imshow( cv2.resize(Image_T2 [slice_idx, 1].cpu().numpy(), (256, 256)), cmap="gray")
        axes[i,1].imshow( cv2.resize(Image_ADC[slice_idx, 1].cpu().numpy(), (256, 256)), cmap="gray")
        axes[i,2].imshow( cv2.resize(Image_DWI[slice_idx, 1].cpu().numpy(), (256, 256)), cmap="gray")
        axes[i,3].imshow( cv2.resize(Image_T2 [slice_idx, 1].cpu().numpy(), (256, 256)), cmap="gray")
    
    for i in range(3): axes[0,i].imshow(contour_image, vmin=0, vmax=2, cmap=label_cmap, alpha=0.4)
    for i in range(4): axes[1,i].imshow(contour_image, vmin=0, vmax=2, cmap=label_cmap, alpha=0.4)
    
    # Gland prediction map
    axes[1,0].imshow(preds_T2_softmax [slice_idx,:3].argmax(0)==1, vmin=0, vmax=3, cmap=jet_transp, alpha=0.2)
    axes[1,1].imshow(preds_ADC_softmax[slice_idx,:3].argmax(0)==1, vmin=0, vmax=3, cmap=jet_transp, alpha=0.2)
    axes[1,2].imshow(preds_DWI_softmax[slice_idx,:3].argmax(0)==1, vmin=0, vmax=3, cmap=jet_transp, alpha=0.2)
    axes[1,3].imshow(preds_MP_softmax [slice_idx,:3].argmax(0)==1, vmin=0, vmax=3, cmap=jet_transp, alpha=0.2)

    # Cancer probability map
    axes[1,0].imshow(preds_T2_softmax [slice_idx,2:4].sum(axis=0), vmin=0.4, vmax=1, cmap=jet_transp, alpha=0.6)
    axes[1,1].imshow(preds_ADC_softmax[slice_idx,2:4].sum(axis=0), vmin=0.4, vmax=1, cmap=jet_transp, alpha=0.6)
    axes[1,2].imshow(preds_DWI_softmax[slice_idx,2:4].sum(axis=0), vmin=0.4, vmax=1, cmap=jet_transp, alpha=0.6)
    axes[1,3].imshow(preds_MP_softmax [slice_idx,2:4].sum(axis=0), vmin=0.4, vmax=1, cmap=jet_transp, alpha=0.6)
    axes[1,3].imshow(contour_image, vmin=0, vmax=2, cmap=label_cmap, alpha=0.5)
    for i, modal in enumerate(['T2 & Label', 'ADC & Label', 'DWI & Label', 'T2 only']):
        axes[0, i].set_title(f"{modal}")
        axes[0, i].axis('off')
    for i, modal in enumerate(['T2 prediction', 'ADC prediction', 'DWI prediction', 'mpMRI prediction']):
        axes[1, i].set_title(f"{modal}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)


def visualize_TRUS(Image, Label, preds_softmax, filename):
    """
    Visualizes one TRUS case in three panels:
      - Origin image (grayscale).
      - Ground-truth label overlay (with contours).
      - Prediction overlay: gland segmentation (binary mask for class 1; blue tint)
        combined with cancer probability heatmap (sum of channels 2 and 3; jet colormap).
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    # assume that prediction is 4-channel
    jet = plt.cm.jet
    colors = jet(np.arange(256))
    colors[:int(0.15 * 256), -1] = 0
    jet_transp = LinearSegmentedColormap.from_list("custom_jet", colors)

    colors = [(0, 0, 0, 0), (1, 1, 0, 1), (1, 0, 0, 1)]  # transparent, and yellow, red
    label_cmap = ListedColormap(colors)

    # Select the representative slice (using torch dims)
    slice_idx = Label[:, 2:4].sum(dim=(1, 2, 3)).argmax().item()

    # Process ground-truth label: get contours from the segmentation map
    segmentation_map = Label.argmax(dim=1).cpu().numpy()[slice_idx]
    segmentation_map[segmentation_map > 2] = 2  # cap labels at 2
    mask_combined = np.logical_or(segmentation_map == 1, segmentation_map == 2).astype(np.uint8)
    mask_label2 = (segmentation_map == 2).astype(np.uint8)
    contours_combined, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_label2, _ = cv2.findContours(mask_label2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_label = np.zeros(segmentation_map.shape, dtype=np.uint8)
    cv2.drawContours(contour_label, contours_combined, -1, 1, 1)
    cv2.drawContours(contour_label, contours_label2, -1, 2, 1)

    # Prepare the original image (assumed grayscale from channel 1)
    origin = Image[slice_idx, 1].cpu().numpy()
    origin = cv2.resize(origin, (256, 256))

    # Process predictions:
    # Gland segmentation: binary mask where predicted label (from first 3 channels) equals 1
    gland_overlay = (preds_softmax[slice_idx, :3].argmax(dim=0) == 1).cpu().numpy().astype(np.uint8)
    # Cancer probability: sum of channels 2 and 3 (cancer classes)
    cancer_overlay = preds_softmax[slice_idx, 2:4].sum(dim=0).cpu().numpy()

    # Create a figure with three panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Origin image (displayed in grayscale, no tint)
    axes[0].imshow(origin, cmap="gray")
    axes[0].set_title("Origin", fontsize=12)
    axes[0].axis("off")
    
    # Panel 2: Ground-truth label overlay (contours)
    axes[1].imshow(origin, cmap="gray")
    axes[1].imshow(contour_label, vmin=0, vmax=2, cmap=label_cmap, alpha=0.5)
    axes[1].set_title("Label", fontsize=12)
    axes[1].axis("off")
    
    # Panel 3: Prediction overlay
    # First, overlay gland segmentation (blue tint) then cancer probability heatmap
    axes[2].imshow(origin, cmap="gray")
    axes[2].imshow(gland_overlay, vmin=0, vmax=3, cmap=jet_transp, alpha=0.2)
    axes[2].imshow(cancer_overlay, cmap=jet_transp, alpha=0.6, vmin=0.4, vmax=1)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # Adjust layout to prevent title cutoff
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)




def saveData(pred, origin_t2, save_filename):
    origin_t2_sitk = sitk.ReadImage(origin_t2)
    pred_image = sitk.GetImageFromArray(pred)
    pred_image.SetSpacing(origin_t2_sitk.GetSpacing())
    pred_image.SetDirection(origin_t2_sitk.GetDirection())
    pred_image.SetOrigin(origin_t2_sitk.GetOrigin())
    sitk.WriteImage(pred_image, save_filename)



def visualize_featuremap(Tokens, Image, Label, filename):
    cancer_max_axial_slice = Label[:, 2:4].sum(axis=(1,2,3)).argmax()
    Tokens_slice = Tokens[cancer_max_axial_slice].view(32, 32, 384)
    Tokens_slice_permute = Tokens_slice.permute(2, 0, 1).unsqueeze(0) # Shape: (384, 32, 32)
    Tokens_upsampled = torch.nn.functional.interpolate(Tokens_slice_permute, size=(256, 256), mode='bilinear', align_corners=False)
    Tokens_upsampled = Tokens_upsampled.squeeze(0).permute(1, 2, 0) # Shape: (256, 256, 384)

    
    flat_features = Tokens_upsampled.view(-1, 384).cpu().numpy()  # (256*256, 384)
    flat_label = Label[cancer_max_axial_slice, 1:].sum(axis=0).view(-1).cpu().numpy()

    roi_indices = np.where(flat_label >= 1)[0]  # Get valid indices
    roi_features = flat_features[roi_indices]  # Extract only ROI features

    n_components = 3
    umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
    roi_features_umap = umap_reducer.fit_transform(roi_features)  # (num_roi, 1)

    flat_output = np.zeros((256*256, n_components)) - 9999  # Initialize with zeros
    flat_output[roi_indices] = roi_features_umap  # Assign UMAP-reduced values to ROI
    Tokens_umap = torch.tensor(flat_output).view(256, 256, 3)

    for umap_channel in range(3):
        Tokens_umap_channel = Tokens_umap[:,:,umap_channel]
        Tokens_umap_channel[Tokens_umap_channel == -9999] = flat_output[roi_indices, umap_channel].min()-0.1
        Tokens_umap_channel = (Tokens_umap_channel - Tokens_umap_channel.min()) / (Tokens_umap_channel.max() - Tokens_umap_channel.min())
        Tokens_umap[:,:,umap_channel] = Tokens_umap_channel
    
    jet = plt.cm.jet
    colors = jet(np.arange(256))
    colors[:1, -1] = 0
    jet_transp = LinearSegmentedColormap.from_list("custom_jet", colors)
    
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5): ax[i].imshow( cv2.resize(Image[cancer_max_axial_slice, 0].cpu().numpy(), (256,256)), cmap='gray')
    ax[1].imshow(Label[cancer_max_axial_slice,1:].sum(axis=0).cpu()>0, vmax=2, cmap=jet_transp, alpha=0.5)
    ax[1].imshow(Label[cancer_max_axial_slice,2:].sum(axis=0).cpu()>0, vmax=1, cmap=jet_transp, alpha=0.5)
    for i in range(3): ax[i+2].imshow(Tokens_umap[:,:,i], alpha=0.5, cmap=jet_transp)
    plt.savefig(filename)
    plt.close()
