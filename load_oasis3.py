import nibabel as nib
import sys 
data_folder="/well/win-fmrib-analysis/users/lhw539/oasis3/data/"
example_file="OAS30869_MR_d1691_anat_T1_brain_to_MNI_lin.nii.gz"
img = nib.load(data_folder+example_file)
data=img.get_fdata()
print(type(data))
print(data.shape)
