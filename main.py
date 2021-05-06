from unet_architecture import UnetArchitecture
from preprocess_data import PreprocessData

# creating training and testing dataset
preprocess_data_obj = PreprocessData(verbose=True)
preprocess_data_obj.create_train_or_test_data('train')
preprocess_data_obj.create_train_or_test_data('test')

# creating network architecture
unet_model_obj = UnetArchitecture(preprocess_data_obj.img_rows, preprocess_data_obj.img_cols, verbose=True)



