import torch
import torch.nn.functional as F
import random


def OneBatchTraining_seg(Image, Label, MODEL, criterion, optimizer, pos=None, modal=None):
    '''
    Perform a single batch training for a segmentation model.
    Args:
        Image (x), tensor: The input images for the training batch with shape [small_batchsize, channel(3), img_size, img_size].
        Label (y), tensor: The ground truth labels for the training batch with shape [small_batchsize, img_class, 256, 256].
            This tensor should contain label data for segmentation.
        MODEL, torch.nn.Module: The segmentation model to be trained.
        criterion, torch.nn: The loss function used for training.
        optimizer, torch.optim: The optimizer used for training.
    
    Returns: The loss value for the current training batch.
    '''
    # Forward pass
    if pos is not None and modal is not None:
        preds = MODEL(Image, pos, modal)
    elif pos is not None:
        preds = MODEL(Image, pos)
    else:
        preds = MODEL(Image)
    
    # Convert labels to the categorical format
    Label_argmax = Label.argmax(axis=1).unsqueeze(1)
    loss = criterion(preds, Label_argmax)
    if MODEL.training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def OneBatchTraining_seg_contrastive(Image, Label, MODEL, criterion, optimizer, pos=None, modal=None, contrastive_alpha=0.15, max_contrastive_pairs = 30, distant_negative=False):
    '''
    Perform a single batch training for a segmentation model.
    Args:
        Image (x), tensor: The input images for the training batch with shape [small_batchsize, channel(3), img_size, img_size].
        Label (y), tensor: The ground truth labels for the training batch with shape [small_batchsize, img_class, 256, 256].
            This tensor should contain label data for segmentation.
        MODEL, torch.nn.Module: The segmentation model to be trained.
        criterion, torch.nn: The loss function used for training.
        optimizer, torch.optim: The optimizer used for training.
    
    Returns: The loss value for the current training batch.
    '''
    # Forward pass
    # if pos is not None and modal is not None:
    #     preds = MODEL(Image, pos, modal)
    # elif pos is not None:
    #     preds, features = MODEL(Image, pos, modal, return_features=True)
    # else:
    #     preds = MODEL(Image)
    preds, features = MODEL(x=Image, pos=pos, modal=modal, return_features=True)
    Label_argmax = Label.argmax(axis=1).unsqueeze(1)
    loss = criterion(preds, Label_argmax)

    # Contrastive Loss
    total_contrastive_loss  = 0.0
    anchor_patches, comparison_patches, pair_labels = generate_cancer_contrastive_pairs(Label, upper_threshold=62, lower_threshold=2, distant_negative=distant_negative)
    anchor_patches, comparison_patches, pair_labels = undersampling_cancer_contrastive_pairs(anchor_patches, comparison_patches, pair_labels, n_max=max_contrastive_pairs)
    # anchor_patches: [ [slice_pos, y_pos, x_pos], ... ]


    if len(anchor_patches) > 0:
        for anchor, target, pair_label in zip(anchor_patches, comparison_patches, pair_labels):
            anchor_feature = MODEL(x=features[anchor[0], :, anchor[1], anchor[2]].unsqueeze(0), forward_head=True).squeeze(0)
            target_feature = MODEL(x=features[target[0], :, target[1], target[2]].unsqueeze(0), forward_head=True).squeeze(0)
            total_contrastive_loss  += contrastive_loss(anchor_feature, target_feature, pair_label)
        total_contrastive_loss /= len(anchor_patches)

    final_loss = loss * (1-contrastive_alpha ) + total_contrastive_loss * contrastive_alpha
    # Convert labels to the categorical format
    if MODEL.training:
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
    return final_loss.item()

def OneBatchInference_seg(Image, Label, MODEL, criterion, dice_metric, args, pos=None, modal=None):
    """
    Perform a single batch inference for a segmentation model, including loss and dice score calculation.
    Args:
        Image (x), Tensor: The input images for the inference batch with shape [small_batchsize, 3, 256, 256].
        Label (y), Tensor: The ground truth labels for the inference batch with shape [small_batchsize, 1, 256, 256].
        MODEL, torch.nn.Module: The segmentation model used for inference.
        criterion, torch.nn: The loss function used for evaluating performance.
        dice_metric, function: A function to calculate the Dice Similarity Coefficient (DSC) for segmentation performance evaluation.

    Returns: list: A list containing the loss value followed by the mean Dice scores for each class.
    """
    with torch.no_grad():  # Disables gradient calculation to save memory and computations

        if pos is not None and modal is not None:
            preds = MODEL(Image, pos, modal)
        elif pos is not None:
            preds = MODEL(Image, pos)
        else:
            preds = MODEL(Image)

            
        # Compute loss
        Label_argmax = Label.argmax(axis=1).unsqueeze(1)
        loss = criterion(preds, Label_argmax)

        # Dice
        one_hot_labels = F.one_hot(Label.argmax(axis=1), num_classes=args.nClass)
        one_hot_labels = one_hot_labels.permute(3, 0, 1, 2).float()        
        one_hot_preds  = F.one_hot(preds.argmax(axis=1), num_classes=args.nClass)
        one_hot_preds  = one_hot_preds.permute(3, 0, 1, 2).float()
        dice_scores    = dice_metric(y_pred=one_hot_preds, y=one_hot_labels)
        mean_val_dice  = torch.nanmean(dice_scores, dim=1).detach().cpu()
        
    return [loss.item()] + mean_val_dice.tolist()

def OneBatchTraining_fusion(Tokens_T2, Tokens_ADC, Tokens_DWI, Label, MODEL_Fusion, criterion, optimizer):
    '''
    Perform a single batch training for a segmentation model.
    Args:
        Image (x), tensor: The input images for the training batch with shape [small_batchsize, channel(3), img_size, img_size].
        Label (y), tensor: The ground truth labels for the training batch with shape [small_batchsize, img_class, 256, 256].
            This tensor should contain label data for segmentation.
        MODEL, torch.nn.Module: The segmentation model to be trained.
        criterion, torch.nn: The loss function used for training.
        optimizer, torch.optim: The optimizer used for training.
    
    Returns: The loss value for the current training batch.
    '''
    # Forward pass
    preds = MODEL_Fusion(Tokens_T2, Tokens_ADC, Tokens_DWI)
    
    # Convert labels to the categorical format
    Label_argmax = Label.argmax(axis=1).unsqueeze(1)
    loss = criterion(preds, Label_argmax)
    if MODEL_Fusion.training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def OneBatchInference_fusion(Tokens_T2, Tokens_ADC, Tokens_DWI, Label, MODEL_Fusion, criterion, dice_metric, args):
    """
    Perform a single batch inference for a segmentation model, including loss and dice score calculation.
    Args:
        Image (x), Tensor: The input images for the inference batch with shape [small_batchsize, 3, 256, 256].
        Label (y), Tensor: The ground truth labels for the inference batch with shape [small_batchsize, 1, 256, 256].
        MODEL, torch.nn.Module: The segmentation model used for inference.
        criterion, torch.nn: The loss function used for evaluating performance.
        dice_metric, function: A function to calculate the Dice Similarity Coefficient (DSC) for segmentation performance evaluation.

    Returns: list: A list containing the loss value followed by the mean Dice scores for each class.
    """
    with torch.no_grad():  # Disables gradient calculation to save memory and computations
        preds = MODEL_Fusion(Tokens_T2, Tokens_ADC, Tokens_DWI)
        # Compute loss
        Label_argmax = Label.argmax(axis=1).unsqueeze(1)
        loss = criterion(preds, Label_argmax)

        # Dice
        one_hot_labels = F.one_hot(Label.argmax(axis=1), num_classes=args.nClass)
        one_hot_labels = one_hot_labels.permute(3, 0, 1, 2).float()        
        one_hot_preds  = F.one_hot(preds.argmax(axis=1), num_classes=args.nClass)
        one_hot_preds  = one_hot_preds.permute(3, 0, 1, 2).float()
        dice_scores    = dice_metric(y_pred=one_hot_preds, y=one_hot_labels)
        mean_val_dice  = torch.nanmean(dice_scores, dim=1).detach().cpu()
    return [loss.item()] + mean_val_dice.tolist()


def tensor_shuffle(Images, Labels, device, pos=None, modal=None, shuffle=True):
    """
    Objectives: This function shuffles data from multiple patients, ensuring that data from several patients are mixed within each batch.

    Args:
        Images (Tensor): A tensor containing the images with shape [batch_size, channels, height, width].
        Labels (Tensor): A tensor containing the labels with shape [batch_size, ...] where ... represents the dimensionality of the label.
        pos (Tensor): A tuple containing the positional information for the images (axis-position, maximum-axial slices).
        modal (Tensor): A tensor containing the modality information for the images (0/1/2/3 = T2/ADC/DWI/TRUS).
    Returns:
        tuple: A tuple containing the shuffled images and labels tensors.
    """
    indices = torch.randperm(Images.size(0))  # Generate a permutation of indices
    Data = [tensor.to(device).float() for tensor in (Images, Labels, pos, modal) if tensor is not None]
    if shuffle:
        return tuple([data[indices].float().to(device) for data in Data])
    return tuple(Data)

def getPatchTokens_TRUS(MODEL, Images, Positions, args):
    Tokens = []
    for idx in range(0, Images.shape[0], args.small_batchsize):
        Image, Posit = Images[idx:idx+args.small_batchsize, :, :, :], Positions[idx:idx+args.small_batchsize, :]
        with torch.no_grad():
            Image_Reduced = MODEL.ChannelReducer(Image) #
            Token = MODEL.forward_features_pos(Image_Reduced, Posit)
        Tokens.append(Token['x_norm_patchtokens'].detach().cpu())
    Tokens = torch.vstack(Tokens)
    return Tokens

def getPatchTokens(MODEL, Images, Positions, args):
    Tokens = []
    for idx in range(0, Images.shape[0], args.small_batchsize):
        Image, Posit = Images[idx:idx+args.small_batchsize, :, :, :], Positions[idx:idx+args.small_batchsize, :]
        with torch.no_grad():
            Token = MODEL.forward_features_pos(Image, Posit)
        Tokens.append(Token['x_norm_patchtokens'].detach().cpu())
    Tokens = torch.vstack(Tokens)
    return Tokens

def generate_cancer_contrastive_pairs(Label, upper_threshold=60, lower_threshold=4, distant_negative=False):
    CancerMap = Label[:,2].unfold(1,8,8).unfold(2, 8, 8).sum(axis=(3,4))
    anchor_patches, comparison_patches, pair_labels = [], [], []
    slice_idx, y_idx, x_idx = torch.where(CancerMap >= upper_threshold)
    for slice_pos, y_pos, x_pos in zip(slice_idx, y_idx, x_idx):
        
        positive_y, positive_x = torch.where(CancerMap[slice_pos, y_pos-1:y_pos+2, x_pos-1:x_pos+2] >= upper_threshold)
        for p_y, p_x in zip(positive_y, positive_x):
            if p_y == 1 and p_x == 1:  # skip anchor patch
                continue
            anchor_patches.append([slice_pos, y_pos, x_pos])
            comparison_patches.append([slice_pos, y_pos-1+p_y, x_pos-1+p_x])
            pair_labels.append(1)  # Positive relationship

        # Negative patch one more distance
        if distant_negative:
            negative_y, negative_x = torch.where(CancerMap[slice_pos, y_pos-1:y_pos+2, x_pos-1:x_pos+2] <= lower_threshold)
            negative_y = ((negative_y-1) * 2) + 1
            negative_x = ((negative_x-1) * 2) + 1     
            for n_y, n_x in zip(negative_y, negative_x):
                if CancerMap[slice_pos, y_pos-1+n_y, x_pos-1+n_x].item() <= lower_threshold:
                    anchor_patches.append([slice_pos, y_pos, x_pos])
                    comparison_patches.append([slice_pos, y_pos-1+n_y, x_pos-1+n_x])
                    pair_labels.append(-1)  # Negative relationship
        else: # Negative patch just next to the anchor
            negative_y, negative_x = torch.where(CancerMap[slice_pos, y_pos-1:y_pos+2, x_pos-1:x_pos+2] <= lower_threshold)
            for n_y, n_x in zip(negative_y, negative_x):
                anchor_patches.append([slice_pos, y_pos, x_pos])
                comparison_patches.append([slice_pos, y_pos-1+n_y, x_pos-1+n_x])
                pair_labels.append(-1)  # Negative relationship

    return anchor_patches, comparison_patches, pair_labels

def undersampling_cancer_contrastive_pairs(anchor_patches, comparison_patches, pair_labels, n_max=30):
    # pair_labels에 따라 인덱스 분리
    positive_indices = [i for i, label in enumerate(pair_labels) if label == 1]
    negative_indices = [i for i, label in enumerate(pair_labels) if label == -1]

    # 더 작은 수의 샘플 수를 기준으로 샘플링
    min_samples = min(len(positive_indices), len(negative_indices), n_max)

    # 랜덤 샘플링을 통해 각 리스트에서 동일한 수의 샘플 선택
    sampled_positive_indices = random.sample(positive_indices, min_samples)
    sampled_negative_indices = random.sample(negative_indices, min_samples)

    # 최종적으로 선택된 인덱스를 합치고, 해당하는 패치 정보 추출
    final_indices = sampled_positive_indices + sampled_negative_indices
    random.shuffle(final_indices)  # 선택된 인덱스를 섞어서 순서에 의한 편향 방지

    # 최종적으로 샘플링된 패치 정보 추출
    final_anchor_patches = [anchor_patches[i] for i in final_indices]
    final_comparison_patches = [comparison_patches[i] for i in final_indices]
    final_pair_labels = [pair_labels[i] for i in final_indices]
    return final_anchor_patches, final_comparison_patches, final_pair_labels


def contrastive_loss(feature1, feature2, label, margin=0.5):
    # feature1, feature2: extracted features from anchor and target, label: 1 (positive) -1 (negative)
    cos_sim = F.cosine_similarity(feature1, feature2, dim=0)
    if label == 1: # Positive pair: similarity to be close to 1
        loss = 1 - cos_sim
    else: # Negative pair: similarity to be less than margin
        loss = F.relu(cos_sim - margin)
    return loss