import SimpleITK as sitk
import numpy as np
#
# img = sitk.ReadImage('B2_CESAG.dcm.nii')
# img_array = sitk.GetArrayFromImage(img)
# print(img_array.max(),img_array.min(),np.shape(img_array))
#
# label = sitk.ReadImage('B2_Label.nii')
# label_array = sitk.GetArrayFromImage(label)
# print(label_array.max(),label_array.min(),np.shape(label_array))

a = np.array()

print(isinstance(a, np.array))