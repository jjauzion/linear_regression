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
parser.add_argument("data", type=str, help="Path to data file")
parser.add_argument("-o", "--output", type=str, default="model.pkl", help="File where to save the model after training")
parser.add_argument("--y_col", type=str, choices=["first", "last"], default="last", help="Position of the Y column")
parser.add_argument("--scaler", type=str, choices=["standard", "minmax", "identity"], default="standard", help="Define the scaler to be used to sclae the training data set")
parser.add_argument("--no_header", action="store_true", help="The csv data file has no header (ie: 1st line is data)")
parser.add_argument("--data_augmentation", action="store_true", help="Will add the square value of X last column of the train data set for 2nd degree polynomal fit")
parser.add_argument("-i", "--nb_iter", type=check_positive, default=500, help="number of iteration for the training")
parser.add_argument("-lr", "--learning_rate", type=check_positive_float, default=0.05, help="number of iteration for the training")
parser.add_argument("-f", "--force", action="store_true", help="number of iteration for the training")
parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], default=1,
                    help="verbose mode: 0 = no print ; 1 = print result ; 2 = plot result")
args = parser.parse_args()

if Path(args.output).is_file():
    overwrite = "y" if args.force else None
    while overwrite != "y" and overwrite != "n":
        overwrite = input("File '{}' already exist. Do you want to overwrite it ? (y/n)\n".format(args.output))
    if overwrite == "n":
        exit(0)
model = LinearRegression()
try:
    model.load_data_from_csv(args.data, y_col=args.y_col, remove_header=False if args.no_header else True, data_augmentation=args.data_augmentation)
except (IsADirectoryError, PermissionError, FileNotFoundError, FileExistsError, NotADirectoryError) as err:
    print("Error: can't load data from '{}' because : {}".format(args.data, err))
    exit(0)
model.train(args.nb_iter, args.learning_rate, scaler_type=args.scaler, verbose=args.verbose)
try:
    model.save_model(args.output)
except (IsADirectoryError, PermissionError, FileNotFoundError, FileExistsError, NotADirectoryError) as err:
    print("Error: can't save model to '{}' because : {}".format(args.output, err))
    exit(0)
if args.verbose > 0:
    print("Model saved to {}".format(args.output))


