# Liver segmentation using U-Net architecture


## Creating setup and installing packages
First install anaconda [Anaconda documentation](https://docs.anaconda.com/anaconda/install/). After run on terminal (Linux only)
```bash
bash setup.sh
```

## Data
The data is available in NifTi format <a href='https://www.dropbox.com/s/8h2avwtk8cfzl49/ircad-dataset.zip?dl=0'>here</a>. 
This dataset consists of 20 medical examinations in 3D, we have the source image as well as a mask of segmentation of the liver for each of these examinations. We use the nibabel library (http://nipy.org/nibabel/) to read associated images and masks.

To use this dataset you must put masks (in the file name will have "liver") and original images (in the file name will have "orig") in separated folder. If you use setup.sh the necessary folders wil be created in order for you just fill it with .nii files.
