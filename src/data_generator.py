import numpy as np
import csv


def test(coef, brand):
    return coef[0] + coef[1] * 1 + coef[2] * j


def generate(coef, nb_of_exprience):
    """

    :param coef:
    :param nb_of_exprience:
    :return:
    """
    coef = ([2, 2, 2], [3, 3, 3])
    data = np.array(
        np.concatenate((np.zeros((nb_of_exprience, 2)), np.random.randint(0, len(coef), (nb_of_exprience, 1))), axis=1),
        dtype="float64")
    np.apply_over_axes(test(coef, ))
