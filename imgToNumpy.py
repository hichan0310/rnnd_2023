import cv2
from filter import binary_filter
import numpy as np
from PIL import Image


def img_array(path, size:tuple[int, int], padding=0):
    arr = [[0 for _ in range(200 + padding * 2)] for _ in range(padding)]
    arr += list(map(lambda a: [0] * padding + list(map(lambda b: 1 if b else 0, a)) + [0] * padding,
                   np.array(
                       Image.fromarray(
                           255 - binary_filter(
                               cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                           )
                       ).resize(size).convert('1'))))
    arr += [[0 for _ in range(200 + padding * 2)] for _ in range(padding)]
    return arr