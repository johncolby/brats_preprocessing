# BraTS Preprocessing

## Installation

```bash
pip install git+https://github.com/johncolby/brats_preprocessing
```

## Usage

```python
import brats_preprocessing.brats_preprocessing as bp

zip_path = '/path/to/<accession>.zip'
model_path = '/path/to/model.Rdata'

mri = bp.tumor_study(zip_path, model_path)

mri.setup()
mri.classify_series()
mri.preprocess()
```

## Inputs

These tools expect an `<accession>.zip` file as input. The archive should contain a single study directory, which should in turn contain series directories, which should in turn contain DICOM files. For example, if I unzip `11111111.zip` into a directory named `11111111`, then the structure should look like:

```
+-11111111/
  |
  +-a8651ef7f70c8cba45e3141f4e3bef17/
  | |
  | +-06b758b1335988a09ee4cb7af3678b98/
  | | |
  | | +-ae01489d59eed67281e13df5ed992af0.dcm
  | | |
  | | +-ad08d6483cb7b02865b3b05b8003a170.dcm
  | | |
  | | +-... <additional files>
  | |
  | +-49e1e56731aac53ea1205e1f080b5c70/
  | | |
  | | +-268f45ba1d77340ff0fe7c4c283e6284.dcm
  | | |
  | | +-fb60f2e00e74a0ddc7e0f5bcaad68df6.dcm
  | | |
  | | +-... <additional files>
  | |
  | +-bfa267d60ff3e8873518773b934f99fc/
  | | |
  | | +-15bce104269e2c8f8cdf274845277c08.dcm
  | | |
  | | +-c081ec04dcade7a2643a62186374111a.dcm
  | | |
  | | +-... <additional files>
  ...
```

Alternatively, if `air_download` is installed, you may use the `download()` method to access the AIR API directly.

```python
mri = bp.tumor_study(acc='11111111')
mri.setup()
mri.download(URL='https://air.<domain>.edu/api/', cred_path='/path/to/air_login.txt')
```

## Series classification

A small but important step in processing clinical data involves identifying the desired input series. For a given application/context, this can be done easily by training a small classifier to identify series type (e.g. axial 3D T1, FLAIR, etc.) based on DICOM header data (study code, TR, TE, series description, etc.). We do that here by loading a custom pre-trained `model.Rdata` file. See URL for details.

Alternatively, series directory paths may be manually added like:

```python
mri.add_paths(['/path/to/flair', 
               '/path/to/t1', 
               '/path/to/t1ce', 
               '/path/to/t2'])
```

## Preprocessing pipeline

Implemented with `nipype`.

1. DICOM files are converted to NIfTI (`dcm2niix`).
1. T1 processing.
    1. Skull strip to get `T1_brain` and `T1_brain_mask` (`BET`).
    1. Register `T1_brain` to MNI template space to get `T1_brain_mni` and `t1_brain_mni.mat` (`FLIRT`).
    1. Apply `t1_brain_MNI.mat` transform to `T1_brain_mask` (`FLIRT`).
    1. Apply final mask in MNI space (`fslmaths`).
1. Non T1 processing, e.g. FLAIR.
    1. Register to T1 but only save `.mat` (`FLIRT`).
    1. Concatenate FLAIR-to-T1 and T1-to-MNI transformation matrices (`convert_xfm`).
    1. Apply concatenated `.mat` to register FLAIR to MNI in a single resampling step (`FLIRT`).
    1. Apply final mask in MNI space (`fslmaths`).
1. Multi-channel bias field correction (`FAST`).
1. Merge into single 4D file (`fslmerge`).
1. Change data type to 16 bit integer (`fslmaths`).
1. Pad voxel dimensions and reorient to BraTS template space (`FLIRT`).

## Notes on the BraTS template

MNI and BraTS template systems are different.

|                  | MNI                | BraTS              |
| -                | -                  | -                  |
| Voxel size       | 1 x 1 x 1          | 1 x 1 x 1          |
| Voxel dimensions | 182 x 218 x 182    | 240 x 240 x 155    |
| Origin           | `[90, -126, -72]`  | `[0, 239, 0]`      |
| Orientation      | LAS (radiological) | LPS (neurological) |

Calculate voxel offsets needed to match origins between BraTS and MNI.

```R
90  + (240 - 182) / 2 # x = 119
126 + (240 - 218) / 2 # y = 137
72  + (155 - 182) / 2 # z = 58.5
```

Can generate a minimal BraTS orientation template *de novo* like:

```bash
fslcreatehd 240 240 155 1 1 1 1 1 119 137 59 4 brats_ref
fslorient -setsformcode 1 brats_ref
fslorient -forceneurological brats_ref
fslswapdim brats_ref x -y z brats_ref_reorient
```

This is included as `brats_ref_reorient.nii.gz`

To convert from MNI to BraTS template space (pad voxel dimensions and reorient), we can use FSL FLIRT:

```bash
flirt -in t1_brain_flirt -ref brats_ref_reorient -applyxfm -usesqform -out output

```