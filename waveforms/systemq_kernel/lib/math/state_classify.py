import numpy as np


def nearest_classify(S, params):
    """多能级 singleshot 区分

    Parameters:
        S: complex-array 复振幅

        params: dict
          c0, c1, c2, ... : 不同能级的中心

    Return:
        states: int-array with the same shape as S
    """

    centers = []
    for i in range(10):
        c_i = params.get(f'c{i}', None)
        if c_i is None:
            break
        else:
            centers.append(c_i)
    centers = np.asarray(centers)
    return np.argmin(np.abs([S - cent for cent in centers]), axis=0)


def circular_region_classify(S, params):
    """多能级 circular 区分

    Parameters:
        S: complex-array 复振幅
        params: dict
          c0, c1, c2, ... : 不同能级的中心
          r0, r1, r2, ... : 不同能级的半径

    Return:
        states: int-array with the same shape as S
    """

    raise NotImplementedError
