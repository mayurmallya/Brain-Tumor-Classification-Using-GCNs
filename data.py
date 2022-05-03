import os
import numpy as np
import torch
import torch.utils.data as udata
import SimpleITK as sitk
import pandas as pd
from augment import augment_data


def resample(image, spacing, size):
    # Create the reference image
    reference_origin = np.zeros(image.GetDimension())
    reference_direction = np.identity(image.GetDimension()).flatten()
    reference_image = sitk.Image(size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(spacing)
    reference_image.SetDirection(reference_direction)

    # Transform which maps from the reference_image to the current image (output-to-input)
    transform = sitk.AffineTransform(image.GetDimension())
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)

    # Modify the transformation to align the centers of the original and reference image
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    centering_transform = sitk.TranslationTransform(image.GetDimension())
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # Using the linear interpolator
    image_rs = sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0)
    return image_rs



class Radpath(udata.Dataset):
    def __init__(self, csv_file, data_path, shuffle):
        self.image_path = data_path
        self.df = pd.read_csv(csv_file)
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label_dict = {'G': 0, 'O': 1, 'A': 2}
        label = label_dict[self.df.loc[idx, 'class']]
        dataID = self.df.loc[idx, 'CPM_RadPath_2019_ID']

        # read niigz file
        T1 = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_t1.nii.gz'))
        T1ce = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_t1ce.nii.gz'))
        T2 = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_t2.nii.gz'))
        FLAIR = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_flair.nii.gz'))

	# resize
        x_size, y_size, z_size = T1.GetSize()
        input_size = [128, 128, 128]
        spacing = [x_size / input_size[0], y_size / input_size[1], z_size / input_size[2]]
        T1 = resample(T1, spacing=spacing, size=input_size)
        T1ce = resample(T1ce, spacing=spacing, size=input_size)
        T2 = resample(T2, spacing=spacing, size=input_size)
        FLAIR = resample(FLAIR, spacing=spacing, size=input_size)

        # convert to one batch of ndarray
        T1 = sitk.GetArrayFromImage(T1).astype(np.float32)
        T1ce = sitk.GetArrayFromImage(T1ce).astype(np.float32)
        T2 = sitk.GetArrayFromImage(T2).astype(np.float32)
        FLAIR = sitk.GetArrayFromImage(FLAIR).astype(np.float32)
        image = np.stack((T1, T1ce, T2, FLAIR), 0)

        # tensor
        data = torch.from_numpy(image)

        return data, label



class Radpath_test(udata.Dataset):
    def __init__(self, csv_file, data_path):
        self.image_path = data_path
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        dataID = self.df.loc[idx, 'CPM_RadPath_2020_ID']
        
        # read niigz file
        T1 = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_t1.nii.gz'))
        T1ce = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_t1ce.nii.gz'))
        T2 = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_t2.nii.gz'))
        FLAIR = sitk.ReadImage(os.path.join(self.image_path, dataID, dataID+'_flair.nii.gz'))

	# resize
        x_size, y_size, z_size = T1.GetSize()
        input_size = [128, 128, 128]
        spacing = [x_size / input_size[0], y_size / input_size[1], z_size / input_size[2]]
        T1 = resample(T1, spacing=spacing, size=input_size)
        T1ce = resample(T1ce, spacing=spacing, size=input_size)
        T2 = resample(T2, spacing=spacing, size=input_size)
        FLAIR = resample(FLAIR, spacing=spacing, size=input_size)

        # convert to one batch of ndarray
        T1 = sitk.GetArrayFromImage(T1).astype(np.float32)
        T1ce = sitk.GetArrayFromImage(T1ce).astype(np.float32)
        T2 = sitk.GetArrayFromImage(T2).astype(np.float32)
        FLAIR = sitk.GetArrayFromImage(FLAIR).astype(np.float32)
        image = np.stack((T1, T1ce, T2, FLAIR), 0)

        # tensor
        data = torch.from_numpy(image)

        return data, dataID
