import numpy as np
import os
import sys
from data_transform.utils import *


class Transform(object):
    def __init__(self, aug_list):
        """
        :param iter_time: time you want to iteration to choose a random opt
        :type iter_time: int
        """
        self.aug_list = aug_list

    def do_aug(self, aug, rgb, depth):
        output = aug(rgb, depth)
        return output[0], output[1]

    def random_aug(self, rgb, depth=None, iter_time=None):
        """
        :param iter_time: set iter_time to -1 to use all the affine methods
        :type iter_time: int
        :return: image, mask
        """
        # if self.fixed_mask == None and depth == None:
        #     raise ValueError('self.fixed mask and mask should not both be None')
        # elif depth == None:
        #     depth = self.fixed_mask
        # if not iter_time:
        #     iter_time = self.iter_time
        # if iter_time == -1:
        #     iter_time = len(self.aug_list)
        # applied_aug_list = np.random.choice(self.aug_list, size=iter_time, replace=False, p=self.prob_list)
        applied_aug_list = np.random.choice(self.aug_list)
        for aug in applied_aug_list:
            rgb, depth = self.do_aug(aug, rgb, depth)
        return rgb, depth


def run_transform(rgb, depth):
    aug_list = [random_whether, random_gamma, blur, random_brightness, hue_saturation_value, rgb_shift, clahe,
                color_transform, affine_transform, rotate15, vertical_flip, horizontal_flip]
    trans = Transform(aug_list, iter_time=5)
    return trans.random_aug(rgb, depth)
