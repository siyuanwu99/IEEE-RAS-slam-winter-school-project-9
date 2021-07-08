import numpy as np
import tensorflow as tf
import sklearn
import os


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)
    return tf.keras.backend.mean(l_depth)


def grad_loss_function(y_true, y_pred, theta=0.1):
    l_grad = ...
    return tf.keras.backend.mean(l_grad)


def ssim_loss_function(y_true, y_pred):
    l_ssim = ...
    return tf.keras.backend.mean(l_ssim)
