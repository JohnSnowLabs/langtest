def save_format(format, save_dir, df_report):
    if format == "dict":
        if save_dir is None:
            raise ValueError('You need to set "save_dir" parameter for this format.')

        df_report.to_json(save_dir)

    elif format == "excel":
        if save_dir is None:
            raise ValueError('You need to set "save_dir" parameter for this format.')
        df_report.to_excel(save_dir)
    elif format == "html":
        if save_dir is None:
            raise ValueError('You need to set "save_dir" parameter for this format.')
        df_report.to_html(save_dir)
    elif format == "markdown":
        if save_dir is None:
            raise ValueError('You need to set "save_dir" parameter for this format.')
        df_report.to_markdown(save_dir)
    elif format == "text" or format == "csv":
        if save_dir is None:
            raise ValueError('You need to set "save_dir" parameter for this format.')
        df_report.to_csv(save_dir)
    else:
        raise ValueError(
            f'Report in format "{format}" is not supported. Please use "dataframe", "excel", "html", "markdown", "text", "dict".'
        )
