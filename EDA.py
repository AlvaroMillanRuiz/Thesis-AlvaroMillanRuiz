import nibabel as nib
from matplotlib import pyplot as plt
from glob import glob
import os
from tqdm import tqdm

seg = r'/data/projects/TMOR/data/OsloPreprocessed/OsloPreprocessed/Subject16/seg_16.nii.gz'
nii = nib.load(seg)
vol = nii.get_fdata()
idx = vol.sum(axis=0).sum(axis=1).argmax()
plt.imshow(vol[:, idx, :], cmap='gray')
plt.show()

len(vol.sum(0).sum(1))
vol.sum(0).sum(1)[140]
idx = vol.sum(axis=0).sum(axis=1)


# seg = r"/data/projects/TMOR/data/OsloPreprocessed/OsloPreprocessed/Subject01/seg_1.nii.gz"
rgx = r'/data/projects/TMOR/data/StanfordPreprocessed/StanfordPreprocessed/Mets_121/seg.nii.gz'
nii = nib.load(rgx)
vol = nii.get_fdata()
idx = vol.sum(axis=0).sum(axis=1).argmax()
plt.imshow(vol[:, idx, :], cmap='viridis')
plt.show()

np.unique(vol)

seg = '/data/projects/TMOR/data/OsloPreprocessed/OsloPreprocessed/*'
files = glob(seg)
for file in files:
    mri = seg + '/seg_1.nii.gz'
    nii = nib.load(mri)
    vol = nii.get_fdata()
    idx = vol.sum(axis=0).sum(axis=1).argmax()
    plt.imshow(vol[:, idx, :], cmap='gray')
    plt.show()

import os
import numpy as np
# # # # # OSLO # # # # # # # # # # # # # # # # shift Alt E
directory = '/data/projects/TMOR/data/OsloPreprocessed/OsloPreprocessed/*/seg_*.nii.gz'
oslo_dir = '/data/projects/TMOR/oslo_dir/{}_seg.png'
os.makedirs(os.path.dirname(oslo_dir), exist_ok=True)
#for subdir, dir, files in os.walk(directory):
for file in tqdm(glob(directory)):
    file_dir = os.path.dirname(file)
    subjid = file_dir.split(os.sep)[-1]
    mris = sorted(glob(os.path.join(file_dir, '*.nii.gz')))
    mris = mris[1:] + mris[:1]

    fig, axes = plt.subplots(2, 2)
    axes = axes.reshape(-1)
    for mri, ax in zip(mris, axes):
        nii = nib.load(mri)
        vol = nii.get_fdata()
        if mri == file:
            idx = vol.sum(0).sum(1).argmax()
            cmap_name = 'viridis'
        else:
            cmap_name = 'gray'
        mri_slice = vol[:, idx, :]
        ax.imshow(np.rot90(mri_slice), cmap=cmap_name)
        ax.set_title(os.path.basename(mri))
    fig.suptitle(f'ID: {subjid} - index: {idx} - n_tumor_voxels: {np.sum(mri_slice)}')
    fig.savefig(oslo_dir.format(subjid))
    plt.close(fig)

#
for file in glob(directory):
    nii = nib.load(file)
    vol = nii.get_fdata()
    file_dir = os.path.dirname(file)
    subjid = file_dir.split(os.sep)[-1]
    print(subjid, len(np.unique(vol)) - 1)


# # # # # STANDFORD # # # # # # # # # # # # # # # # shift Alt E
directory = '/data/projects/TMOR/data/StanfordPreprocessed/StanfordPreprocessed/*/seg.nii.gz'
standford_dir = '/data/projects/TMOR/STANDFORD_dir/{}_seg.png'
os.makedirs(os.path.dirname(standford_dir), exist_ok=True)

for file in tqdm(glob(directory)):
    file_dir = os.path.dirname(file)
    subjid = file_dir.split(os.sep)[-1]
    mris = sorted(glob(os.path.join(file_dir, '*.nii.gz')))
    mris = mris[2:] + mris[:2]
    #### visualize the data ###
    fig, axes = plt.subplots(2, 3)
    axes = axes.reshape(-1)
    for mri, ax in zip(mris, axes):
        nii = nib.load(mri)
        vol = nii.get_fdata()
        if mri == file:
            idx = vol.sum(0).sum(1).argmax()
            cmap_name = 'viridis'
        else:
            cmap_name = 'gray'
        mri_slice = vol[:, idx, :]
        ax.imshow(np.rot90(mri_slice), cmap=cmap_name)
        ax.set_title(os.path.basename(mri))
    fig.suptitle(f'ID: {subjid} - index: {idx} - n_tumor_voxels: {np.sum(mri_slice)}')
    fig.savefig(standford_dir.format(subjid))
    plt.close(fig)

for file in glob(directory):
    nii = nib.load(file)
    vol = nii.get_fdata()
    file_dir = os.path.dirname(file)
    subjid = file_dir.split(os.sep)[-1]
    print(subjid, len(np.unique(vol)) - 1)




directory = '/data/projects/TMOR/data/OsloPreprocessed/OsloPreprocessed/*/seg_*.nii.gz'
oslo_dir = '/data/projects/TMOR/oslo_dir/{}_seg.png'
os.makedirs(os.path.dirname(oslo_dir), exist_ok=True)
#for subdir, dir, files in os.walk(directory):
for file in tqdm(glob(directory)):
    file_dir = os.path.dirname(file) #reading the multiple files of the Brain Metastases
    subjid = file_dir.split(os.sep)[-1] #getting the name from the patients
    mris = sorted(glob(os.path.join(file_dir, '*.nii.gz')))
    mris = mris[1:] + mris[:1] # placing the file with the segmnented brain metastasis first
    #putting all the images from a single patient together for better visualization
    fig, axes = plt.subplots(2, 2)
    axes = axes.reshape(-1)
    for mri, ax in zip(mris, axes):
        nii = nib.load(mri) #read the mris
        vol = nii.get_fdata() #get the volume from the images
        if mri == file:
            idx = vol.sum(0).sum(1).argmax() #aggregate over the two axis to find the tumors by the voxel values
            cmap_name = 'viridis'
        else:
            cmap_name = 'gray'
        mri_slice = vol[:, idx, :]
        ax.imshow(np.rot90(mri_slice), cmap=cmap_name)
        ax.set_title(os.path.basename(mri))
    fig.suptitle(f'ID: {subjid} - index: {idx} - n_tumor_voxels: {np.sum(mri_slice)}')
    fig.savefig(oslo_dir.format(subjid))
    plt.close(fig)


num_tumors = []
for file in glob(directory):
    nii = nib.load(file)
    vol = nii.get_fdata()
    file_dir = os.path.dirname(file)
    subjid = file_dir.split(os.sep)[-1]
    print(subjid, len(np.unique(vol)) - 1)
    num_tumors.append(len(np.unique(vol)) - 1)

np.unique(vol)