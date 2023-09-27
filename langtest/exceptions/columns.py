class ColumnNameError(Exception):
    """ColumnNameError class is used to raise an exception
    when the column name is not found in the dataset."""

    def __init__(
        self,
        supported_columns,
        given_columns,
        message="\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {supported_columns} \
                    \nOr classifiy the features and target columns from {given_columns}",
    ):
        self.message = message.format(
            supported_columns=supported_columns, given_columns=given_columns
        )
        super().__init__(self.message)
