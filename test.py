# %%
import nibabel
# %%
img = nibabel.load("dataset/test/masks/ircad_e19_liver.nii.gz")
# %%
data = img.get_fdata()
# %%
nibabel.viewers.OrthoSlicer3D(data).show()
# %%
import SimpleITK as sitk
read_1 = sitk.ReadImage("D:\\Code\\DeepLearning\\liver_segmantation_Unet\\dataset\\test\\originals\\ircad_e20_orig.nii.gz")
read_2 = sitk.ReadImage("D:\\Code\\DeepLearning\\liver_segmantation_Unet\\dataset\\test\\masks\\ircad_e20_liver.nii.gz")
# %%
array_1 = sitk.GetArrayFromImage(read_1)
array_2 = sitk.GetArrayFromImage(read_2)
# %%
array_1.shape, array_2.shape
# %%
import numpy as np
array_1 = array_1[...,np.newaxis]
# %%
array_2 = array_2[...,np.newaxis]
# %%
array_1.shape, array_2.shape
# %%
# %%
array_1
# %%
from make_tf_dataset import image_to_tf_data
# %%
img_path = "D:\\Code\\DeepLearning\\liver_segmantation_Unet\\Segmentation_Rigid_Training\\Training\\OP1\\Masks\\img_01_class.png"
mask_path = "D:\\Code\\DeepLearning\\liver_segmantation_Unet\\Segmentation_Rigid_Training\\Training\\OP1\\Raw\\img_01_raw.png"
data_1 = image_to_tf_data(img_path=img_path,mask_path=mask_path,class_name='m')
# %%
import tensorflow as tf
tf_writer = tf.io.TFRecordWriter('train.record')
tf_writer.write(data_1.SerializeToString())
tf_writer.close()
# %%
raw_dataset = tf.data.TFRecordDataset('train.record')
raw_dataset
# %%
raw_dataset
# %%
dataset_12 = tf.data.Dataset.from_tensor_slices((array_1,array_2))
dataset_12
# %%
dataset_12 = dataset_12.prefetch(1)
# %%
