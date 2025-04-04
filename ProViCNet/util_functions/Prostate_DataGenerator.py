from icecream import ic
import glob
import os
import numpy as np
import nibabel as nib
import albumentations as A
import SimpleITK as sitk
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

class US_MRI_Generator(Dataset):
    def __init__(self, imageFileName, glandFileName, cancerFileName, modality,
                       cancerTo2=False, Augmentation=False,
                       filter_background_prob={'TRUS':0.20, 'MRI':1.00}, img_size=256, Image_Only=False,
                       return_modal=False, nChannel=3):
        self.imageFileName = imageFileName
        self.glandFileName = glandFileName
        self.cancerFileName = cancerFileName
        self.modality = modality
        self.Image_Only = Image_Only
        self.return_modal = return_modal
        self.nChannel = nChannel
        if nChannel % 2 == 0:
            raise ValueError("nChannel should be even number")
        
        self.transformAug = A.Compose([
            A.ElasticTransform(alpha=0.2, sigma=15, alpha_affine=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(0, 15), p=0.3),
            A.Affine(scale=None, translate_percent=None, rotate=0, shear=2, p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3)
        ], additional_targets={'label1': 'mask', 'label2': 'mask'})
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
        ], additional_targets={'label1': 'mask', 'label2': 'mask'})

        self.cancerTo2 = cancerTo2
        self.Augmentation = Augmentation
        self.filter_background_prob = filter_background_prob
        self.resizer = A.Compose([
            A.Resize(img_size, img_size)
        ])
    
    # def _loadImage(self, idx):
    #     image = sitk.GetArrayFromImage(sitk.ReadImage(self.imageFileName[idx]))/255
    #     if self.Image_Only:
    #         return image
    #     gland = sitk.GetArrayFromImage(sitk.ReadImage(self.glandFileName[idx]))
    #     cancer = sitk.GetArrayFromImage(sitk.ReadImage(self.cancerFileName[idx]))
    #     return image.astype(np.float32), gland.astype(np.float32), cancer.astype(np.float32)

    def _loadImage(self, idx):
        # Read the image files
        image = sitk.ReadImage(self.imageFileName[idx])
        gland = sitk.ReadImage(self.glandFileName[idx])
        cancer = sitk.ReadImage(self.cancerFileName[idx])
        
        # Ensure the origin and direction match
        gland.SetOrigin(image.GetOrigin())
        gland.SetDirection(image.GetDirection())
        
        cancer.SetOrigin(image.GetOrigin())
        cancer.SetDirection(image.GetDirection())
        
        # Convert images to numpy arrays and normalize image
        image_array = sitk.GetArrayFromImage(image) / 255.0
        gland_array = sitk.GetArrayFromImage(gland)
        cancer_array = sitk.GetArrayFromImage(cancer)
        
        # Convert to float32
        image_array = image_array.astype(np.float32)
        gland_array = gland_array.astype(np.float32)
        cancer_array = cancer_array.astype(np.float32)
        
        if self.Image_Only:
            return image_array
        
        return image_array, gland_array, cancer_array


    def _cancerTo2(self, cancer):
        cancer[cancer > 1] = 1
        return cancer

    def _cancerTo3(self, cancer):
        cancer[cancer > 2] = 2
        return cancer

    def filterBackground(self, gland, filter_background_prob=.1):
        prostate_present = np.any(gland == 1, axis=(1, 2))
        non_prostate_indices = np.where(~prostate_present)[0]
        num_to_select = int(len(non_prostate_indices) * filter_background_prob)
        selected_indices = np.random.choice(non_prostate_indices, size=num_to_select, replace=False)
        slicesIndex = sorted(np.where(prostate_present)[0].tolist() + selected_indices.tolist())
        return slicesIndex
    
    def __len__(self):
        return len(self.imageFileName)

    def imageTransform(self, image, gland, cancer):
        imageAugmented, glandAugmented, cancerAugmented = [], [], []
        for imageSlice, glandSlice, labelSlice in zip(image, gland, cancer): # albumentations를 사용한 augmentation 적용
            if self.Augmentation:
                augmented_slide = self.transformAug(image=imageSlice, label1=glandSlice, label2=labelSlice)
                imageSlice, glandSlice, labelSlice = augmented_slide['image'], augmented_slide['label1'], augmented_slide['label2']
            imageSlice = self.resizer(image=imageSlice)['image']

            imageAugmented.append(imageSlice)
            glandAugmented.append(glandSlice)
            cancerAugmented.append(labelSlice)

        imageAugmented  = np.stack(imageAugmented)
        glandAugmented  = np.stack(glandAugmented)
        cancerAugmented = np.stack(cancerAugmented)
        return imageAugmented, glandAugmented, cancerAugmented

    def filterBackground(self, gland, filter_background_prob=.1):
        prostate_present = np.any(gland == 1, axis=(1, 2))
        non_prostate_indices = np.where(~prostate_present)[0]
        num_to_select = int(len(non_prostate_indices) * filter_background_prob)
        selected_indices = np.random.choice(non_prostate_indices, size=num_to_select, replace=False)
        slicesIndex = sorted(np.where(prostate_present)[0].tolist() + selected_indices.tolist())
        return slicesIndex

    def getNChannel(self, image, gland, cancer, filter_background=1.0, nChannel=3):
        if image.shape[0] < nChannel * 2:
            padding = (nChannel // 2, nChannel // 2)
            image = np.pad(image, (padding, (0, 0), (0, 0)), mode='constant')
            gland = np.pad(gland, (padding, (0, 0), (0, 0)), mode='constant')
            cancer = np.pad(cancer, (padding, (0, 0), (0, 0)), mode='constant')
        
        if filter_background == 1.0:
            slicesIndex = list(range(len(image)))
        else:
            slicesIndex = self.filterBackground(gland, filter_background)
        # nChannel = nChannel//2
        image3Channel = []
        gland3Channel = []
        cancer3Channel = []
        for idx in slicesIndex: # start_idx, end_idx should be three consecutive, but (0 <= x < 256)
            if idx-nChannel//2 < 0:
                idx += nChannel//2
            elif idx+nChannel//2 >= len(image):
                idx -= nChannel//2
            start_idx, end_idx = idx - nChannel//2, idx + nChannel//2 + 1
            image3Channel.append(image[start_idx:end_idx, :, :])
            gland3Channel.append(gland[idx, :, :])
            cancer3Channel.append(cancer[idx, :, :])
        image3Channel = np.stack(image3Channel)
        gland3Channel = np.stack(gland3Channel)
        cancer3Channel = np.stack(cancer3Channel)
        return image3Channel, gland3Channel, cancer3Channel

    def normalizeImage(self, image, gland=None):
        if gland is not None:
            MR_parameters = image[ gland == 1 ].mean(), image[ gland == 1 ].std()
            image = (image - MR_parameters[0]) / MR_parameters[1]
        else: image = (image - image.mean()) / image.std()
        return image
    
    def __getitem__(self, idx):
        if self.Image_Only:
            image, self._loadImage(idx)
            image, _, _ = self.imageTransform(image, image, image)
            image = self.normalizeImage(image, gland)
            image, _, _ = self.getNChannel(image, image, image, self.filter_background_prob, nChannel=self.nChannel)
            if self.return_modal:
                return image, self.modality
            return image
        elif self.Image_Only == False:
            image, gland, cancer = self._loadImage(idx)
            cancer.max()
            cancer = self._cancerTo2(cancer) if self.cancerTo2 else self._cancerTo3(cancer)
            if 'PICAI' in self.imageFileName[idx] and self.cancerTo2 == False:
                cancer[cancer == 1] = 2

            image = self.normalizeImage(image, gland)
            image, gland, cancer = self.imageTransform(image, gland, cancer)
            background_prob = self.filter_background_prob['MRI'] if self.modality in ['MRI', 'ADC', 'T2', 'DWI'] else self.filter_background_prob['TRUS']
            image, gland, cancer = self.getNChannel(image, gland, cancer, filter_background=background_prob, nChannel=self.nChannel)
        
            if self.return_modal:
                return image, gland, cancer, self.modality
            return image, gland, cancer


def collate_prostate(batch):
    Images  = torch.vstack([ torch.from_numpy(batch[i][0]) for i in range(len(batch))])        
    Gland  = torch.vstack([ torch.from_numpy(batch[i][1]) for i in range(len(batch))])
    Cancer = torch.vstack([ torch.from_numpy(batch[i][2].astype(np.float32)) for i in range(len(batch))])
    Gland[ Cancer == 1] = 0
    Backg  = (~((Gland==1) + (Cancer==1))).float()
    Labels = torch.stack( [Backg, Gland, Cancer] ).permute(1,0,2,3)
    return Images.float(), Labels.long()

def collate_prostate_position(batch):
    Images  = torch.vstack([ torch.from_numpy(batch[i][0]) for i in range(len(batch))])        
    Gland  = torch.vstack([ torch.from_numpy(batch[i][1]) for i in range(len(batch))])
    Cancer = torch.vstack([ torch.from_numpy(batch[i][2].astype(np.float32)) for i in range(len(batch))])
    
    Axials = []
    for i in range(len(batch)):
        Axials += [(j, len(batch[i][0])) for j in range(len(batch[i][0]))]

    Gland[ Cancer == 1] = 0
    Backg  = (~((Gland==1) + (Cancer==1))).float()
    Labels = torch.stack( [Backg, Gland, Cancer] ).permute(1,0,2,3)
    return Images.float(), Labels.long(), torch.tensor(Axials).float()

def collate_prostate_position_CS(batch):
    Images  = torch.vstack([ torch.from_numpy(batch[i][0]) for i in range(len(batch))])        
    Gland  = torch.vstack([ torch.from_numpy(batch[i][1]) for i in range(len(batch))])
    Cancer = torch.vstack([ torch.from_numpy(batch[i][2].astype(np.float32)) for i in range(len(batch))])
    
    Axials = []
    for i in range(len(batch)):
        Axials += [(j, len(batch[i][0])) for j in range(len(batch[i][0]))]

    Gland[ Cancer == 1] = 0
    Gland[ Cancer >= 2] = 0
    Backg  = (~((Gland==1) + (Cancer==1) + (Cancer==2))).float()
    Labels = torch.stack( [Backg, Gland, (Cancer==1).float(), (Cancer >= 2).float()] ).permute(1,0,2,3)
    return Images.float(), Labels.long(), torch.tensor(Axials).float()

def collate_prostate_position_modal(batch):
    Images  = torch.vstack([ torch.from_numpy(batch[i][0]) for i in range(len(batch))])        
    Gland  = torch.vstack([ torch.from_numpy(batch[i][1]) for i in range(len(batch))])
    Cancer = torch.vstack([ torch.from_numpy(batch[i][2].astype(np.float32)) for i in range(len(batch))])
    Axials = []
    for i in range(len(batch)):
        Axials += [(j, len(batch[i][0])) for j in range(len(batch[i][0]))]

    modalities = ['ADC', 'DWI', 'T2', 'TRUS']  # 가능한 모달리티 종류
    modality_indices = {mod: index for index, mod in enumerate(modalities)}
    Modalities = []
    for i in range(len(batch)):
        modality = batch[i][3]
        num_slices = len(batch[i][0])  # 현재 배치 아이템의 슬라이스 수
        if modality in modality_indices:
            modality_index = modality_indices[modality]
            modality_one_hot = torch.zeros((num_slices, len(modalities)), dtype=torch.float32)
            modality_one_hot[:, modality_index] = 1
            Modalities.append(modality_one_hot)
    Modalities = torch.vstack(Modalities)

    Gland[ Cancer == 1] = 0
    Backg  = (~((Gland==1) + (Cancer==1))).float()
    Labels = torch.stack( [Backg, Gland, Cancer] ).permute(1,0,2,3)
    return Images.float(), Labels.long(), torch.tensor(Axials), Modalities


def getData(Image_path, Gland_path, Label_path, Modality, file_extensions):
    """
    This function collects image, gland, and label file paths for a given modality, organizing them into a DataFrame.
    It is designed to handle data where file names include a modality-specific identifier, such as '84216_001_trus.nii.gz'.
    
    Parameters:
    - Image_path: Path to the directory containing the image files.
    - Gland_path: Path to the directory containing the gland annotation files.
    - Label_path: Path to the directory containing the cancer label files.
    - Modality: A string representing the imaging modality (e.g., 'MRI', 'US') used in file naming.
    - file_extensions: A dictionary specifying the postfix to be added to each filename for images, glands, and labels.
    
    Returns:
    - A pandas DataFrame with columns for [image paths, gland paths, label paths, modality, and patient ID].
    """

    Image, Gland, Label = [], [], []
    Modal, patID = [], []

    img = [i.split(file_extensions['Image_name'])[0] for i in os.listdir(Image_path)] # patientID_modality.nii.gz
    gld = [i.split(file_extensions['Gland_name'])[0] for i in os.listdir(Gland_path)] # patientID_modality_prostate_label.nii.gz
    can = [i.split(file_extensions['Cancer_name'])[0] for i in os.listdir(Label_path)] # patientID_modality_roi_bxconfirmed_label.nii.gz
    
    intersection_filenames = set(img) & set(gld) & set(can)
    for filename in sorted(intersection_filenames):
        Image.append(Image_path + filename + file_extensions['Image_name'])
        Gland.append(Gland_path + filename + file_extensions['Gland_name'])
        Label.append(Label_path + filename + file_extensions['Cancer_name'])
        Modal.append(Modality)
        patID.append(filename)

    Dataset = pd.DataFrame({'Image':Image, 'Gland':Gland, 'Cancer':Label, 'Modality':Modal, 'PatientID':patID})
    Dataset.set_index('PatientID', inplace=True)
    return Dataset

def aggregate(df):
    return pd.concat(df.values(), axis=0).reset_index(drop=True)

def align_pred_with_ref(pred_tensor, ref_sitk):
    pred_np = pred_tensor.cpu().numpy()
    pred_sitk = sitk.GetImageFromArray(pred_np)
    pred_sitk.SetSpacing(ref_sitk.GetSpacing())
    pred_sitk.SetOrigin(ref_sitk.GetOrigin())
    pred_sitk.SetDirection(ref_sitk.GetDirection())
    return pred_sitk

def resize_pred_to_label(pred_sitk, label_sitk):
    # Ensure the orientation matches
    if pred_sitk.GetDirection() != label_sitk.GetDirection():
        pred_sitk = sitk.Resample(pred_sitk, label_sitk, sitk.Transform(), sitk.sitkNearestNeighbor)
    
    # Resample pred_sitk to match label_sitk
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(label_sitk)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resized_pred = resample.Execute(pred_sitk)
    return resized_pred

def input_verification_crop_or_pad(image,size,physical_size):
    """
    Calculate target size for cropping and/or padding input image
    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)
    - physical_size: target size in mm (z, y, x)
    Either size or physical_size must be provided.
    Returns:
    - shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    - size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    """
    # input conversion and verification
    if physical_size is not None:
        # convert physical size to voxel size (only supported for SimpleITK)
        if not isinstance(image, sitk.Image):
            raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
        # spacing_zyx = list(image.GetSpacing())[::-1]
        spacing_zyx = list(image.GetSpacing()) # (0.5,0.5,3.0)
        size_zyx = [length/spacing for length, spacing in zip(physical_size, spacing_zyx)]
        size_zyx = [int(np.round(x)) for x in size_zyx] # size after cropping

        if size is None:
            # use physical size
            size = size_zyx
        else:
            # verify size
            # print("size",size)
            # print("size_zyx",size_zyx)
            if size != size_zyx:
                raise ValueError(f"Size and physical size do not match. Size: {size}, physical size: "
                                 f"{physical_size}, spacing: {spacing_zyx}")
    if isinstance(image, sitk.Image):
        # determine shape and convert convention of (z, y, x) to (x, y, z) for SimpleITK
        shape = image.GetSize()
        # size = list(size)[::-1]
        size = list(size)
    else:
        # determine shape for numpy array
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
        size = list(size) # size before cropping
    rank = len(size)
    assert rank <= len(shape) <= rank + 1, \
        f"Example size doesn't fit image size. Got shape={shape}, output size={size}"
    return shape, size

def crop_or_pad(image, size, physical_size, crop_only = False,center_of_mass = None):
    """
    Resize image by cropping and/or padding
    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)       -->(256,256,3.0)
    - physical_size: target size in mm (z, y, x)  -->(256*0.5,256*0.5,num_slice*3.0)
    Either size or physical_size must be provided.
    Returns:
    - resized image (same type as input)
    """
    # input conversion and verification
    #print('center_of_mass',center_of_mass)
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # set identity operations for cropping and padding
    # print('----size------',size)
    rank = len(size)
    padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None) for _ in range(rank)]

    # for each dimension, determine process (cropping or padding)
    if center_of_mass is not None:
        desired_com = [int(x / 2) for x in size]
        displacement = [(x-y) for x,y in zip(desired_com,center_of_mass)]
        for i in range(rank):
            # if shape[i] < size[i]:
            if  int(center_of_mass[i] - np.floor((size[i]) / 2.))<0 or int(center_of_mass[i] + np.floor((size[i]) / 2.))>=256:
                if crop_only:
                    continue
                padding[i][0] = max((size[i] - shape[i]) // 2 + displacement[i],0)
                padding[i][1] = max(size[i] - shape[i] - padding[i][0],0)
                # padding[i][0] = np.maximum(np.floor((np.array(size[i]) - np.array(shape[i])) / 2).astype(int) - displacement[i], 0)
                # padding[i][1] = size[i] - padding[i][0] - shape[i]
            # else:
            idx_start = max(int(center_of_mass[i] - np.floor((size[i]) / 2.)),0)
            idx_end = min(int(center_of_mass[i] + np.floor((size[i]) / 2.)),shape[i])
            #print('idx_start',idx_start)
            #print('idx_end',idx_end)
            slicer[i] = slice(idx_start, idx_end)
    else:
        for i in range(rank):
            if shape[i] < size[i]:
                if crop_only:
                    continue
                # set padding settings
                padding[i][0] = (size[i] - shape[i]) // 2
                padding[i][1] = size[i] - shape[i] - padding[i][0]
            else:
                # create slicer object to crop image
                idx_start = int(np.floor((shape[i] - size[i]) / 2.))
                idx_end = idx_start + size[i]
                slicer[i] = slice(idx_start, idx_end)

    # print("slicer",slicer)

    # crop and/or pad image
    if isinstance(image, sitk.Image):
        #print('padding',padding)
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        return pad_filter.Execute(image)[tuple(slicer)]
    else:
        return np.pad(image[tuple(slicer)], padding)


def main():


    ## Load data directly
    DataPath = '/home/sosal/Data/'
    Image_path = os.path.join(DataPath, 'TRUS/*')
    Gland_path = os.path.join(DataPath, 'TRUS_Prostate_Label/*')
    Label_path = os.path.join(DataPath, 'TRUS_ROI_Bxconfirmed_Label/*')
    
    Image_file = glob.glob(Image_path) # assume list of Image, Gland, Label is same!
    Gland_file = glob.glob(Gland_path)
    Label_file = glob.glob(Label_path)
    
    # Assume all of the modality of files is TRUS
    modality = ['TRUS' for i in range(len(Image_file))]

    Generator = US_MRI_Generator(
        Image_file, Gland_file, Label_file, modality, 
        cancerTo2=True, Augmentation=True, filter_background_prob=.1) #, transformUS, transformMRI)
    
    a, b, c = Generator.__getitem__(10)
    ic(a.shape, b.shape, c.shape)
    ic(a.max(), a.min(), a.mean())
    ic(b.max(), b.min())
    ic(c.max(), c.min())


if __name__ == "__main__":
    main()
