# slam-winter-schoo
This repo is the code and documents of project 9d for IEEE RAS Winter School on SLAM in Deformable Environments. It has been done by several prospective students from both master program of [Robotics](https://www.tudelft.nl/en/education/programmes/masters/robotics/msc-robotics) at TU Delft and [Autonomous Systems](https://www.dtu.dk/english/education/msc/programmes/autonomous-systems) from DTU.

## Brief Description
This work is to implement a Monocular Depth Estimation network using Tensorflow 2 and it is based on the previous work [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/abs/1812.11941) wit modifications.


## Award
We have been awarded the 3rd Prize (6 groups over 19 groups were awarded prize) on IEEE RAS Winter School on SLAM in Deformable Environments. This repo contains the code and some documents about our project. If you have any problems, feel free to email to weihaoxuan97 or siyuanwu99 at gmail dot com.

## Tasks

- [x] Finish the network. You need to configure environment, revise the dataset folder, adjust parameters, and fill this gap. The used functions may include: “tensorflow.keras.applications.DenseNet169”, “tf.keras.layers.Conv2D”, “tf.keras.layers.UpSampling2D”, “tf.keras.layers.Concatenate”, “tf.keras.layers.LeakyReLU”, and “tf.keras.Model”.
- [x] Improve the loss function.
- [x] Add some data augmentation.



## Usage 
The source code is in /src
```console
foo@bar:~$ cd ./src
```

```python
# train
python train.py
```

```python
# evaluate depth metric
python evaluate.py
```

```python
# test and visualization
python test.py --input 'kitti-examples/*.png'
```
