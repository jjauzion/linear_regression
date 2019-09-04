import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import argparse


def exp_decrease(x, x_origin=150000):
    """
    generate price following exponential decrease function based on age (in years) and the new_value of the car
    :param x:
    :param x_origin: new value of the car in euro
    :return: value of the car
    """
    # value = x_origin * math.exp(-x / 10000)
    value = x_origin + 0.05 * x[0] + 0.1 * (x[0] ** 2)
    return random.gauss(value, value / 4)


def generate(nb_of_exprience, mileage_max, new_value):
    """

    :param nb_of_exprience:
    :param new_value: new price of the car
    :return:
    """
    age_vect = np.random.ranf((nb_of_exprience, 1)) * mileage_max
    data = np.vstack((age_vect.T, np.apply_along_axis(exp_decrease, 1, age_vect, x_origin=new_value)))
    return data.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="data/generated.csv", help="output file")
    args = parser.parse_args()
    file = Path(args.output)
    df = generate(1000, 250000, 200000)
    plt.scatter(x=df[:, 0], y=df[:, 1])
    try:
        with file.open(mode='w', encoding='utf-8') as fp:
            np.savetxt(fp, df, delimiter=",")
        print("Dataset save to '{}'".format(file))
    except (IsADirectoryError, PermissionError, FileNotFoundError, FileExistsError, NotADirectoryError) as err:
        print("Error: can't save data to '{}' because : {}".format(file, err))
        exit(0)
    plt.show()
