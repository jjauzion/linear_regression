import pandas as pd
from pathlib import Path


class Data:

    def __init__(self):
        print("hello")
        self.df = None

    def create_df_from_csv(self, csv_file):
        """
        Create a Padnas df from a csv file
        :param csv_file: csv file path
        """
        with Path(csv_file).open(mode='r', encoding='utf-8') as file:
            self.df = pd.read_csv(file)
