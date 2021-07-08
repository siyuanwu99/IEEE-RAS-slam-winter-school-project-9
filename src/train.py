import numpy as np
import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt
import time
from loss import *
from model import *
import data_augmentation as transform

from skimage import io
from zipfile import ZipFile

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# python train.py --data nyu
parser = argparse.ArgumentParser(description='My first complete deep learning code')  # Input parameters
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')  # The batch size of the training network
parser.add_argument('--max_depth', type=int, default=1000,
                    help='The maximal depth value')  # The max depth of the images
parser.add_argument('--data', default="nyu", type=str, help='Training dataset.')  # A default train dataset
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')  # GPU number
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate')

args = parser.parse_args()  # Add input as parameters

## Data augmentation !!!
# optional: rotate15, rotate60, rotate45, vertical_flip, horizontal_flip, random_crop, affine_transfrom
data_augmentation = transform.rotate60

# def _parse_function(filename, label):
#     # Read images from disk
#
#     image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
#     depth_decoded = tf.image.decode_jpeg(tf.io.read_file(label))
#     # print(filename.shape)
#     # print(label.shape)
#     # image_decoded = cv2.imread(filename)
#     # depth_decoded = cv2.imread(label)
#     print(image_decoded.shape)
#
#     ## Data augmentation !!!
#     # optional: rotate15, rotate60, rotate45, vertical_flip, horizontal_flip, random_crop, affine_transfrom
#     image_new, depth_new = transform.rotate60(np.array(image_decoded), np.array(depth_decoded))
#
#     # image_new = []
#     # depth_new = []
#     # for i in range(len(filenames)):
#     #     image_, depth_ = transform.rotate60(np.array(image_decoded[i]), np.array(depth_decoded[i]))
#     #     image_new.append(image_)
#     #     depth_new.append(depth_)
#     # image_new, depth_new = [transform.rotate60(
#     #     np.array(image_decoded[i]), np.array(depth_decoded[i])) for i in range(len(filenames))]
#     # image_new = np.stack(image_new)
#     # depth_new = np.stack(depth_new)
#     shape_depth = (240, 320, 1)
#     depth_resized = tf.image.resize(depth_new, [shape_depth[0], shape_depth[1]])
#
#     # Format
#     rgb = tf.image.convert_image_dtype(image_new, dtype=tf.float32)
#     depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)
#
#     # Normalize the depth values (in cm)
#     depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)
#     return rgb, depth


######################### Prepare Trainning Set ##########################
current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())

## Train_dataset
root = '/home/siyuan/Desktop/slam_winter_school/'  # Please change this root path
csv_file = '../data/test_train.csv'
csv = open(csv_file, 'r').read()
nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))
nyu2_train = sklearn.utils.shuffle(nyu2_train, random_state=0)
filenames = [os.path.join(root, i[0]) for i in nyu2_train]
labels = [os.path.join(root, i[1]) for i in nyu2_train]
length = len(filenames)

image_list = []
depth_list = []
for i in range(length):
    image_decoded = cv2.imread(filenames[i])
    depth_decoded = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
    image_, depth_ = data_augmentation(np.array(image_decoded), np.array(depth_decoded))
    image_ = np.resize(image_, [480, 640, 3])
    depth_ = np.resize(depth_, [240, 320])[:, :, np.newaxis]
    image_list.append(image_)
    depth_list.append(depth_)

image_new = np.stack(image_list)
depth_new = np.stack(depth_list)

# Format
rgb = tf.image.convert_image_dtype(image_new, dtype=tf.float32)
depth = tf.image.convert_image_dtype(depth_new / 255.0, dtype=tf.float32)

# Normalize the depth values (in cm)
depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = tf.data.Dataset.from_tensor_slices((rgb, depth))
dataset = dataset.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
dataset = dataset.repeat()
# dataset = dataset.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
batch_size = args.batch_size  # batch_size from inputs, default value is 2
train_generator = dataset.batch(batch_size=batch_size)

######################### Define Model ##########################

model = DepthEstimate()

######################### Multi-gpu setup ##########################
if args.gpus > 1: model = tf.keras.utils.multi_gpu_model(model, gpus=args.gpus)

######################### Trainning ################################
print('\n\n\n', 'Compiling model..')
model.compile(optimizer=tf.optimizers.Adam(1e-2, lr=args.lr, amsgrad=True), loss=sum2)
# model.summary()
print('\n\n\n', 'Compiling complete')

# Create checkpoint callback
checkpoint_path = "../checkpoints/train_{}/cp.ckpt".format(current_time)
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.fit(train_generator, epochs=args.epochs, steps_per_epoch=length // batch_size, callbacks=[cp_callback])

###########################Save model###############################

model.save(
    os.path.join(root, "model/", "{}_model.h5".format(current_time))
    , include_optimizer=False)
model.save(os.path.join(root, 'models/'), save_format='tf', include_optimizer=False)
