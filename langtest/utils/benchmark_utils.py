import os
import pandas as pd


class Leaderboard:
    """
    Leaderboard class to manage the ranking of the models

    Args:
        path (str): The path to the summary file


    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern to ensure only one instance of the class is created
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path:str, *args, **kwargs) -> None:
        """
        Initialize the Leaderboard class with the summary file
        """
        self.summary = Summary(path, *args, **kwargs)

    def get_score_board(self):
        """
        Get the score board for the models
        """
        df = self.summary.summary_df

        # find the timestamp with the highest score
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp", ascending=False)
        idx = df.groupby(
            ["timestamp", "model", "dataset_name", "split", "test_type", "category"]
        )["score"].idxmax()
        df = df.loc[idx]
        pvt_table = df.pivot_table(
            index=["model"], columns="dataset_name", values="score"
        )
        pvt_table = pvt_table.rename_axis(None, axis=1).reset_index()
        pvt_table = pvt_table.fillna("-")

        # mean column
        pvt_table.insert(1, "Avg", pvt_table.iloc[:, 1:].mean(axis=1))
        pvt_table = pvt_table.sort_values(by="Avg", ascending=False)

        return pvt_table

    def get_score_board_by_tests(self):
        """
        Get the score board for the models by test type
        """

        df = self.summary.summary_df
        pvt_table = df.pivot_table(
            index=["model", "split"], columns=["dataset_name"], values="score"
        )
        # pvt_table.columns = [f"{col[0]}\n{col[1]}" for col in pvt_table.columns]
        # pvt_table = pvt_table.rename_axis(None, axis=1).reset_index()
        pvt_table = pvt_table.fillna("-")

        return pvt_table

    def get_score_board_by_category(self):
        """
        Get the score board for the models by category
        """
        df = self.summary.summary_df
        pvt_table = df.pivot_table(
            index=["model", "category"], columns=["dataset_name"], values="score"
        )
        pvt_table.insert(0, "Avg", pvt_table.mean(axis=1))
        pvt_table = pvt_table.fillna("-")
        pvt_table = pvt_table.rename_axis(None, axis=1).reset_index()

        return pvt_table

    def __repr__(self) -> str:
        return self.summary.summary_df.to_markdown()


class Summary:
    """
    Summary class to manage the summary report
    """
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        """
        Singleton pattern to ensure only one instance of the class is created
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path, *args, **kwargs) -> None:
        """
        Initialize the summary
        """
        self.file_path = path
        self.summary_df: pd.DataFrame = self.load_data_from_file(path, *args, **kwargs)

    def load_data_from_file(self, path: str, *args, **kwargs) -> pd.DataFrame:
        """
        Check if file exists
        """
        try:
            if os.path.exists(path):
                return self.__read_from_csv(path, *args, **kwargs)
            else:
                # Create a new file
                df = pd.DataFrame(columns=self.__default_columns())
                df.to_csv(path, index=False)
                return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {path}")

    def __read_from_csv(self, path: str) -> pd.DataFrame:
        """
        Read data from csv file
        """
        df = pd.read_csv(path)
        return df

    def __default_columns(self):
        """
        Default columns for the summary report
        """
        cols = [
            "timestamp",
            "task",
            "model",
            "hub",
            "category",
            "test_type",
            "dataset_name",
            "split",
            "subset",
            "total_records",
            "success_records",
            "failure_records",
            "score",
        ]
        return cols

    def add_report(
        self,
        generated_results: pd.DataFrame,
    ) -> None:
        """
        Add a new report to the summary
        """

        from datetime import datetime

        # Filter the dataframe for accuracy, fairness and representation
        afr_df = self.__afr(generated_results)
        not_afr_df = self.__not_afr(generated_results)

        # concatenate the dataframes
        temp_summary_df = pd.concat([afr_df, not_afr_df], axis=0)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp_summary_df["timestamp"] = timestamp

        # insert row to the summary df
        self.summary_df = pd.concat([self.summary_df, temp_summary_df], ignore_index=True)

        # Save the summary to the file
        self.save_summary()

    def save_summary(self) -> None:
        """
        Save the summary to the file
        """
        self.summary_df.to_csv(self.file_path, index=False)

    def __afr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataframe for accuracy, fairness and representation
        to be used in the summary report
        """
        df = df[df["category"].isin(["accuracy", "fairness", "representation"])]
        df = df[self.__group_by_cols() + ["actual_result"]]
        df = df.rename(columns={"actual_result": "score"})

        return df

    def __not_afr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataframe for non accuracy, fairness and representation
        to be used in the summary report
        """
        df = df[~df["category"].isin(["accuracy", "fairness", "representation"])]

        grouped = df.groupby(self.__group_by_cols())

        # Filter the columns
        import numpy as np

        total_records = grouped.size().reset_index(name="total_records")
        success_records = grouped["pass"].sum().reset_index(name="success_records")
        score = grouped["pass"].mean().reset_index(name="score")
        failure_records = grouped.apply(
            lambda x: np.size(x["pass"]) - np.sum(x["pass"])
        ).reset_index(name="failure_records")

        # concatenate the dataframes
        result = pd.concat(
            [
                success_records,
                failure_records["failure_records"],
                total_records["total_records"],
                score["score"],
            ],
            axis=1,
        )

        return result

    def __group_by_cols(self):
        """
        Group by columns
        """
        return [
            "category",
            "dataset_name",
            "test_type",
            "model",
            "hub",
            "split",
            "subset",
            "task",
        ]

    @property
    def df(self) -> pd.DataFrame:
        return self.summary_df
