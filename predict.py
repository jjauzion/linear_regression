import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.linear_regression import LinearRegression


def plot_synthesis(model, mileage, price):
    fig = plt.figure()
    plt.scatter(x=model.X_original, y=model.y, label="training Dataset")
    plt.scatter(x=mileage, y=price, c='r', label="Prediction")
    imin = np.argmin(model.X, axis=0)[1]
    imax = np.argmax(model.X, axis=0)[1]
    p1 = np.matmul(model.X[imin], model.weight)
    p2 = np.matmul(model.X[imax], model.weight)
    plt.plot([model.X_original[imin], model.X_original[imax]], [p1, p2], c='g', label="Prediction line")
    plt.legend()
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="Path to saved model pickle")
parser.add_argument("--mileage", type=int, default=-1, help="Mileage of the car you want to price")
args = parser.parse_args()

model = LinearRegression()
loaded = False
if Path(args.model).is_file():
    loaded = model.load_model(args.model)
if not loaded:
    print("--> WARNING: No valid model file found. Prediction will be done with untrained model.")
if args.mileage < 0:
    while 1:
        mileage = input("What mileage is your car ?\n")
        try:
            float(mileage)
        except ValueError as err:
            print("Wrong value: {}".format(err))
        else:
            mileage = float(mileage)
            if mileage >= 0:
                break
            else:
                print("Mileage shall be positive. Got {}".format(mileage))
    print("mileage = {}".format(mileage))
    x = [mileage]
else:
    x = [args.mileage]
price = model.predict(x, verbose=2)
