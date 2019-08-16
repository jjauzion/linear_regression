from src.linear_regression import LinearRegression
import argparse
from pathlib import Path


def check_positive(value):
    try:
        ivalue = int(value)
    except Exception:
        raise argparse.ArgumentTypeError("{} is not a positive int value".format(value))
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is not a positive int value".format(value))
    return ivalue


def check_positive_float(value):
    try:
        ivalue = float(value)
    except Exception:
        raise argparse.ArgumentTypeError("{} is not a positive value".format(value))
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} is not a positive value".format(value))
    return ivalue


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, help="Path to data file")
parser.add_argument("-o", "--output", type=str, default="model/model.pkl", help="File where to save the model after training")
parser.add_argument("--y_col", type=str, choices=["first", "last"], default="last", help="Position of the Y column")
parser.add_argument("--no_header", action="store_true", help="The csv data file has no header (ie: 1st line is data)")
parser.add_argument("-i", "--nb_iter", type=check_positive, default=200, help="number of iteration for the training")
parser.add_argument("-lr", "--learning_rate", type=check_positive_float, default=0.05, help="number of iteration for the training")
args = parser.parse_args()

if not Path(args.data).is_file():
    print("File {} not found.".format(args.data))
    exit(0)
if Path(args.output).is_file():
    overwrite = None
    while overwrite != "y" and overwrite != "n":
        overwrite = input("File {} already exist. Do you want to overwrite it ? (y/n)\n".format(args.output))
    if overwrite == "n":
        exit(0)
model = LinearRegression()
model.load_data_from_csv(args.data, y_col=args.y_col, remove_header=False if args.no_header else True)
model.train(args.nb_iter, args.learning_rate)
model.save_model(args.output)


