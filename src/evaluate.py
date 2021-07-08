import time
from loss import *
from model import *
from train import _parse_function
from train import args
import matplotlib.pyplot as plt

root = '/home/siyuan/Desktop/slam_winter_school/'
checkpoints_path = "../checkpoints/train_07-08-21-16-28/checkpoint"
model_path = "../model/train_07-08-21-16-28_model.h5"

######################### Prepare Test Set ##########################
csv_file_test = '../data/nyu2_test.csv'
csv_test = open(csv_file_test, 'r').read()
nyu2_test = list((row.split(',') for row in (csv_test).split('\n') if len(row) > 0))
nyu2_test = sklearn.utils.shuffle(nyu2_test, random_state=0)
filenames_test = [os.path.join(root, i[0]) for i in nyu2_test]
labels_test = [os.path.join(root, i[1]) for i in nyu2_test]
length_test = len(filenames_test)
dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
dataset_test = dataset_test.shuffle(buffer_size=len(filenames_test), reshuffle_each_iteration=True)
dataset_test = dataset_test.repeat()
dataset_test = dataset_test.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
batch_size = args.batch_size  # batch_size from inputs, default value is 2
test_generator = dataset_test.batch(batch_size=batch_size)

########################## Load Model ################################

model = DepthEstimate()
model.load_weights(checkpoints_path)
model.summary()

# !!! otherwise load the whole model
# new_model = tf.keras.models.load_model(model_path)
# new_model.summary()


########################## Result test ################################
score = model.evaluate(test_generator, steps=10)

print("last score:", score)

######################### Predict a result #############################
image_decoded = tf.image.decode_jpeg(tf.io.read_file('1.jpg'))
rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
rgb = np.expand_dims(rgb, axis=0)
# model = tf.keras.models.load_model('./models/model.h5',custom_objects={'depth_loss_function': depth_loss_function})
result = model.predict(rgb)
# print(result)
image_new = result[0, :, :, 0]
plt.imshow(image_new)
plt.show()