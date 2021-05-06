from os import listdir, path, getcwd
from numpy import array, unique, ndarray, uint8, save, load
from nibabel import load
from skimage.transform import resize

from tqdm import tqdm

class PreprocessData:

    def __init__(self, verbose):
        self.verbose = verbose
        self.img_rows = 256
        self.img_cols = 256

    def create_train_or_test_data(self, subsetType):

        subsetType = str(subsetType).lower()
        if self.verbose:
            print('-' * 70)
            print(f'Creating {subsetType}ing data...')

        mask_path = path.join(getcwd(), f'dataset/{subsetType}/masks')
        orig_path = path.join(getcwd(), f'dataset/{subsetType}/originals')

        mask_img = sorted(listdir(mask_path))
        orig_img = sorted(listdir(orig_path))

        imgs = [] # training images
        masks = [] # training masks (corresponding to the liver)

        for liver, orig in tqdm(zip(mask_img, orig_img), total=len(mask_img)):

            mask_3d = load(path.join(mask_path, liver))
            imag_3d = load(path.join(orig_path, orig))

            for axial_z in range(mask_3d.shape[2]):

                mask_2d = resize(
                    array(mask_3d.get_data()[:, :, axial_z]),
                    (self.img_rows, self.img_cols)
                )  # axial cuts are made along the z axis with undersampling
                image_2d = resize(
                    array(imag_3d.get_data()[:, :, axial_z]),
                    (self.img_rows, self.img_cols)
                )

                if len(unique(mask_2d)) != 1: # if mask_2d contains only 0, it means that there is no liver
                    masks.append(mask_2d)
                    imgs.append(image_2d)

        imgs_orig = ndarray((len(imgs), self.img_rows, self.img_cols), dtype=uint8)
        imgs_mask = ndarray((len(masks), self.img_rows, self.img_cols), dtype=uint8)

        for index, img in enumerate(imgs):
            imgs_orig[index, :, :] = img

        for index, img in enumerate(masks):
            imgs_mask[index, :, :] = img

        save(path.join(getcwd(), f'outputs/{subsetType}_imgs.npy'), imgs_orig)
        save(path.join(getcwd(), f'outputs/{subsetType}_masks.npy'), imgs_mask)

        if self.verbose:
            print(f'\n{subsetType.capitalize()} dataset: saved to outputs folder .npy files with original and masks images.\n')

    @staticmethod
    def load_train_or_test_data(subsetType):
        origs = load(f'{subsetType}_imgs.npy')
        masks = load(f'{subsetType}_masks.npy')
        return origs, masks