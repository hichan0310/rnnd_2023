import numpy as np
from math import pi, sin, cos
from conv import make_conv

grad_range_not_first = pi
grad_rate=0.7


def followline(result: list[tuple[int, int]],
               image: list[list[int]],
               position: tuple[int, int],
               conv_size: int,
               movement: float,
               grad: float = 0,
               grad_range: float = 2 * pi):
    result.append(position)

    conv = make_conv(conv_size, grad_range, grad)
    conv_result = np.array(image)[
                  position[0] - conv_size:position[0] + conv_size + 1,
                  position[1] - conv_size:position[1] + conv_size + 1
                  ] * conv


    for _ in range(position[0] - conv_size // 5, position[0] + conv_size // 5 + 1):
        for __ in range(position[1] - conv_size // 5, position[1] + conv_size // 5 + 1):
            image[_][__] = 0

    num, sum = 0, 0
    for _ in conv_result:
        for __ in _:
            if __ != 0:
                num += 1
                sum += __
    if num == 0:
        return 0

    new_grad = grad + sum / num*grad_rate
    new_position = (
        int(position[0] - movement * sin(new_grad) + 0.5),
        int(position[1] + movement * cos(new_grad) + 0.5)
    )

    return 1 + followline(
        result=result,
        image=image,
        position=new_position,
        conv_size=conv_size,
        movement=movement,
        grad=new_grad,
        grad_range=grad_range_not_first
    )
