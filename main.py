# %%
from unet_architecture import UnetArchitecture
from preprocess_data import PreprocessData
# %%
# creating training and testing dataset
preprocess_data_obj = PreprocessData(verbose=True)
preprocess_data_obj.create_train_or_test_data('train')
preprocess_data_obj.create_train_or_test_data('test')
# %%
# creating network architecture
unet_model_obj = UnetArchitecture(preprocess_data_obj.img_rows, preprocess_data_obj.img_cols, verbose=True)
# %%
preprocess_data_obj.print_path('train')
# %%
train_ds = preprocess_data_obj.load_train_or_test_data('train')
test_ds = preprocess_data_obj.load_train_or_test_data('test')
# %%
train_ds[1].shape
# %%
import tensorflow as tf
train_ds_tf = tf.data.Dataset.from_tensor_slices((train_ds[0], train_ds[1]))
# %%
def preprocess_image(image, label):
    image = tf.reshape(image, [256, 256, 1])
    image = tf.cast(image, tf.float32) / 255.
    label = tf.reshape(label, [256, 256, 1])
    label = tf.cast(label, tf.float32) / 255.
    
    return image, label
# %%
tf_train_data = train_ds_tf.map(
    preprocess_image, 
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
tf_train_data = tf_train_data.batch(64)
tf_train_data = tf_train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
tf_train_data
# %%
unet_model_obj = UnetArchitecture(preprocess_data_obj.img_rows, preprocess_data_obj.img_cols, verbose=True)
# %%
unet_model_obj.model.fit(tf_train_data)
# %%
