from scipy.interpolate import Rbf
import skimage
import skimage.transform
import numpy as np


class PointsRBF:
    def __init__(self, src, dst):
        xsrc = src[:, 0]
        ysrc = src[:, 1]
        xdst = dst[:, 0]
        ydst = dst[:, 1]
        self.rbf_x = Rbf(xsrc, ysrc, xdst, function='linear')
        self.rbf_y = Rbf(xsrc, ysrc, ydst, function='linear')

    def __call__(self, xy):
        x = xy[:, 0]
        y = xy[:, 1]
        xdst = self.rbf_x(x, y)
        ydst = self.rbf_y(x, y)
        return np.transpose([xdst, ydst])


def warpRBF(image, src, dst):
    prbf = PointsRBF(dst, src)
    warped = skimage.transform.warp(image, prbf)
    warped = 255 * warped  # 0..1 => 0..255
    warped = warped.astype(np.uint8)  # convert from float64 to uint8
    return warped
