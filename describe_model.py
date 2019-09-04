from src import linear_regression
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="model file to read (file shall be a pickle format)")
args = parser.parse_args()

model = linear_regression.LinearRegression()
try:
    success = model.load_model(args.model)
except (IsADirectoryError, PermissionError, FileNotFoundError, FileExistsError, NotADirectoryError) as err:
    print("Could not load model because: {}".format(err))
    exit(0)
if success:
    model.describe()
