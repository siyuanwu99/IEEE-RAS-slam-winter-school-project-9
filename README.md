# slam-winter-school


[Dataset](https://studentutsedu-my.sharepoint.com/personal/12514586_student_uts_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F12514586%5Fstudent%5Futs%5Fedu%5Fau%2FDocuments%2Fnyu%5Fdata%2Ezip&parent=%2Fpersonal%2F12514586%5Fstudent%5Futs%5Fedu%5Fau%2FDocuments&originalPath=aHR0cHM6Ly9zdHVkZW50dXRzZWR1LW15LnNoYXJlcG9pbnQuY29tLzp1Oi9nL3BlcnNvbmFsLzEyNTE0NTg2X3N0dWRlbnRfdXRzX2VkdV9hdS9FVTJKNDE1MmVEcE1yN0hYTnZwdmZTSUI5OWpuWWxEUzd3bC13emZFbDNDQjZRP3J0aW1lPWlWeTZ3dWxCMlVn)
approximate 4.1G

# Brief Description
This work is to implement a Monocular Depth Estimation network using Tensorflow 2 and it is based on previous work [High Quality Monocular Depth Estimation via Transfer Learning](https://arxiv.org/abs/1812.11941).


# Award
We have been awarded the 3rd Prize (6 groups over 19 groups were awarded prize) on IEEE RAS Winter School on SLAM in Deformable Environments. This repo contains the code and some documents about our project. If you have any problems, feel free to email to weihaoxuan97 or siyuanwu99 at gmail dot com.

# Tasks

- [x] Finish the network. You need to configure environment, revise the dataset folder, adjust parameters, and fill this gap. The used functions may include: “tensorflow.keras.applications.DenseNet169”, “tf.keras.layers.Conv2D”, “tf.keras.layers.UpSampling2D”, “tf.keras.layers.Concatenate”, “tf.keras.layers.LeakyReLU”, and “tf.keras.Model”.
- [x] Improve the loss function.
- [x] Add some data augmentation.



# Usage 
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
