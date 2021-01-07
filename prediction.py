import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from keras.layers import Layer, Input, Dropout, Conv3D, Activation, add, BatchNormalization, UpSampling3D, \
    Conv3DTranspose, Flatten, concatenate, MaxPooling3D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Network
import cv2
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
import SimpleITK as sitk
import tensorflow as tf

import glob
import matplotlib.image as mpimage
import nibabel as nib
import numpy as np
import datetime
import time
import json
import csv
import sys
import keras

from keras.models import load_model
from keras.models import model_from_json
import json


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        size_increase = [0, 2 * self.padding[0], 2 * self.padding[1], 2 * self.padding[2], 0]
        output_shape = list(s)

        for i in range(len(s)):
            if output_shape[i] == None:
                continue
            output_shape[i] += size_increase[i]

        return tuple(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def DICOM_read_vertical(filePath):

    if filePath[-2:] == "gz":
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(filePath)
        image = reader.Execute()
    elif filePath[-3:] == "nii":
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(filePath)
        image = reader.Execute()
    elif filePath[-4:] == "nrrd":
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(filePath)
        image = reader.Execute()
    else:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(filePath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

    numpy_array = sitk.GetArrayFromImage(image)

    return numpy_array


## Model read
##############
model_path = "F:\\model\\Medical-Image-Synthesis-multiview-Resnet_patch3D\\runs\\20201208-180043-T1-CT_cropped_bias_P9_LR_0.0002_RL_9_DF_64_GF_32_RF_70\\models\\"

json_file =  open(model_path+"G_A2B_model_epoch_360.json",'r')
loded_model_json = json_file.read()
json_file.close()

G_A2B = model_from_json(loded_model_json, custom_objects={'ReflectionPadding3D':ReflectionPadding3D} )
G_A2B.load_weights(model_path+'G_A2B_model_epoch_360.hdf5')


data_path = "F:\\syntheticGANs\\data\\T1-CT_cropped_bias_P9\\trainA\\"
result_path = "F:\\syntheticGANs\\data\\T1-CT_cropped_bias_P9\\trainB\\"
##############

## Prediction
##############

num = 1

input_data = DICOM_read_vertical(data_path + "input_image" + str(num) + ".nii")

real = DICOM_read_vertical(result_path + "simulation" + str(num) + ".nii")

syn = G_A2B.predict(input_data[np.newaxis, :, :, :, np.newaxis])
syn = np.squeeze(syn)




