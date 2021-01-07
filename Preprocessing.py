import numpy as np
import SimpleITK as sitk

trans_image0 = sitk.ReadImage('D:/200_SMA/trans_num0_full.nii')
trans_array0 = sitk.GetArrayFromImage(trans_image0)
max0 = np.max(trans_array0)

for i in range(1250):
    # trans_image = sitk.ReadImage('D:/200_SMA/trans_num'+str(i)+'_full.nii')
    # trans_array = sitk.GetArrayFromImage(trans_image)
    # trans_array = trans_array/np.max(trans_array)

    skull_image = sitk.ReadImage('D:/200_SMA/trans_num'+str(i)+'_full.nii')
    skull_array = sitk.GetArrayFromImage(skull_image)
    spacing = skull_image.GetSpacing()
    origin = skull_image.GetOrigin()
    dimension = skull_image.GetSize()

    # skull_array[skull_array>3000] = 3000
    # skull_array[skull_array<220] = 0
    skull_array = skull_array/max0

    input_array = skull_array
    # input_array = skull_array+trans_array

    input_image = sitk.GetImageFromArray(input_array)
    input_image.SetSpacing(spacing)
    input_image.SetOrigin(origin)

    writer = sitk.ImageFileWriter()
    writer.SetFileName('F:\syntheticGANs\data\T1-CT_cropped_bias_P9/trainB/simulation'+str(i)+'.nii')
    writer.Execute(input_image)

    if i%100 == 0:
        print(i)
