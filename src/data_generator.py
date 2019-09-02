import numpy as np
import math
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import random


def exp_decrease(age, new_value=150000):
    """
    generate price following exponential decrease function based on age (in years) and the new_value of the car
    :param age: value in year
    :param new_value: new value of the car in euro
    :return: value of the car
    """
    value = new_value * math.exp(-age / 100000)
    return random.gauss(value, value / 4)


def generate(nb_of_exprience, mileage_max, new_value):
    """

    :param nb_of_exprience:
    :param new_value: new price of the car
    :return:
    """
    age_vect = np.random.ranf((nb_of_exprience, 1)) * mileage_max
    data = np.vstack((age_vect.T, age_vect.T ** 2, np.apply_along_axis(exp_decrease, 1, age_vect, new_value=new_value)))
    return data


if __name__ == "__main__":
    file = Path("../data/price_exp.csv")
    df = generate(100, 250000, 200000)
    print(df)
    plt.scatter(x=df[0,:], y=df[2,:])
    plt.show()
    with file.open(mode='w', encoding='utf-8') as fp:
        np.savetxt(fp, df, delimiter=",")
