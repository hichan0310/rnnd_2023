import numpy as np
from math import pi, atan

# 사이즈는 1, 2, 3, 4사분면에서 각각 size*size => 총 사이즈는 2*size+1이 반환됨

def make_conv(size:int, grad_range:float, grad:float):
    conv_size=2*size+1
    return np.array([list(map(lambda x: x + 0.00001 if abs(x) <= grad_range / 2 else 0,
                              list(map(lambda x: x if x < pi else x - 2 * pi,
                                       [(0 if x - size == 0 and y - size == 0
                                         else (
                                                  -pi / 2 if x - size == 0 and y - size > 0
                                                  else (
                                                      pi / 2 if x - size == 0 and y - size < 0
                                                      else ((-atan(
                                                          (y - size) / (
                                                                  x - size))) if x - size > 0
                                                            else (pi - atan(
                                                          (y - size) / (
                                                                  x - size))
                                                                  ))
                                                  )
                                              )
                                              - grad + 4 * pi) % (2 * pi)
                                        for x in range(conv_size)])))) for y in
                     range(conv_size)])