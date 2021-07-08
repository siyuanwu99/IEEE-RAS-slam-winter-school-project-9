import numpy as np
import tensorflow as tf
import sklearn
import os


# We are using parameters from DenseDepth

def sum1(y_true, y_pred, w1=1.0, w2=1.0, w3=0.1, maxDepthVal=1000.0 / 10.0):
    return (w1 * depth_loss_function(y_true, y_pred)) + \
           (w2 * grad_loss_function(y_true, y_pred)) + \
           (w3 * ssim_loss_function(y_true, y_pred, maxDepthVal))


def sum2(y_true, y_pred, w1=1.0, w2=1.0, w3=0.1, maxDepthVal=1000.0 / 10.0):
    return (w1 * BerHu_loss(y_true - y_pred)) + \
           (w2 * grad_BerHu_loss_function(y_true, y_pred)) + \
           (w3 * ssim_loss_function(y_true, y_pred, maxDepthVal))


def l1_loss(residual):
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(residual), axis=-1)
    return tf.keras.backend.mean(l_depth)


def BerHu_loss(residual):
    abs_res = tf.keras.backend.abs(residual)
    c = 0.2 * tf.keras.backend.max(abs_res)
    l1 = abs_res
    l2 = 0.5 / c * (tf.keras.backend.square(c) + tf.keras.backend.square(abs_res))
    loss = tf.where(abs_res > c, l2, l1)
    return tf.keras.backend.mean(tf.keras.backend.mean(loss, axis=1))


def depth_loss_function(y_true, y_pred):
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)
    return tf.keras.backend.mean(l_depth)


def grad_loss_function(y_true, y_pred, theta=0.1):
    # compute image edges
    gy_true, gx_true = tf.image.image_gradients(y_true)
    gy_pred, gx_pred = tf.image.image_gradients(y_pred)
    return l1_loss(gy_pred - gy_true) + l1_loss(gx_pred - gx_true)  # Note: distributive law


def grad_BerHu_loss_function(y_true, y_pred):
    # compute image edges
    gy_true, gx_true = tf.image.image_gradients(y_true)
    gy_pred, gx_pred = tf.image.image_gradients(y_pred)
    return BerHu_loss(gy_pred - gy_true) + BerHu_loss(gx_pred - gx_true)


def ssim_loss_function(y_true, y_pred, maxDepthVal):
    l_ssim = tf.keras.backend.clip(0.5 * (1 - tf.image.ssim(y_true, y_pred, maxDepthVal)),
                                   0, 1)
    return tf.keras.backend.mean(l_ssim)






