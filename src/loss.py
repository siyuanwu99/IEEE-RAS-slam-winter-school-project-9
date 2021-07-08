import numpy as np
import tensorflow as tf
import sklearn
import os


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)
    return tf.keras.backend.mean(l_depth)


def grad_loss_function(y_true, y_pred, theta=0.1):
    # compute image edges
    gy_true, gx_true = tf.image.image_gradients(y_true)
    gy_pred, gx_pred = tf.image.image_gradients(y_pred)
    l_grad = tf.keras.backend.mean(
        tf.keras.backend.abs(gy_pred - gy_true) +
        tf.keras.backend.abs(gx_pred - gx_true),
        axis=1
    )
    return tf.keras.backend.mean(l_grad)


def ssim_loss_function(y_true, y_pred, maxDepthVal):
    l_ssim = tf.keras.backend.clip(0.5 * (1 - tf.image.ssim(y_true, y_pred, maxDepthVal)),
                                   0, 1)
    return tf.keras.backend.mean(l_ssim)


def sum_loss_function(y_true, y_pred, w1, w2, w3, maxDepthVal=1000.0 / 10.0):
    return (w1 * depth_loss_function(y_true, y_pred)) + \
           (w2 * grad_loss_function(y_true, y_pred)) + \
           (w3 * ssim_loss_function(y_true, y_pred, maxDepthVal))


def BerHu_loss_function(y_true, y_pred, value):
    c = 0.2 * tf.keras.backend.max(y_true - y_pred)


    return
