from src.linear_regression import LinearRegression
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="model/model.pkl", help="Path to saved model pickle")
args = parser.parse_args()

model = LinearRegression()
loaded = False
if Path(args.model).is_file():
    loaded = model.load_model(args.model)
if not loaded:
    print("--> WARNING: No valid model file found. Prediction will be done with untrained model.")
while(1):
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
price = model.predict([mileage])
print("Estimated price : {}".format(price))
