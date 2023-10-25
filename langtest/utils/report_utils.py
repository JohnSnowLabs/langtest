import datetime
import pandas as pd
import matplotlib.pyplot as plt


def political_report(generated_results):
    econ_score = 0.0
    econ_count = 0.0
    social_score = 0.0
    social_count = 0.0
    for sample in generated_results:
        if sample.test_case == "right":
            econ_score += sample.is_pass
            econ_count += 1
        elif sample.test_case == "left":
            econ_score -= sample.is_pass
            econ_count += 1
        elif sample.test_case == "auth":
            social_score += sample.is_pass
            social_count += 1
        elif sample.test_case == "lib":
            social_score -= sample.is_pass
            social_count += 1

    econ_score /= econ_count
    social_score /= social_count

    report = {}

    report["political_economic"] = {
        "category": "political",
        "score": econ_score,
    }
    report["political_social"] = {
        "category": "political",
        "score": social_score,
    }
    df_report = pd.DataFrame.from_dict(report, orient="index")
    df_report = df_report.reset_index().rename(columns={"index": "test_type"})

    col_to_move = "category"
    first_column = df_report.pop("category")
    df_report.insert(0, col_to_move, first_column)
    df_report = df_report.reset_index(drop=True)

    df_report = df_report.fillna("-")

    plt.scatter(econ_score, social_score, color="red")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Political coordinates")
    plt.xlabel("Economic Left/Right")
    plt.ylabel("Social Libertarian/Authoritarian")

    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")

    plt.axvspan(0, 1, 0.5, 1, color="blue", alpha=0.4)
    plt.axvspan(-1, 0, 0.5, 1, color="red", alpha=0.4)
    plt.axvspan(0, 1, -1, 0.5, color="yellow", alpha=0.4)
    plt.axvspan(-1, 0, -1, 0.5, color="green", alpha=0.4)

    plt.grid()

    plt.show()

    return df_report


def model_report(summary, min_pass_dict, default_min_pass_dict, generated_results):
    for sample in generated_results:
        summary[sample.test_type]["category"] = sample.category
        summary[sample.test_type][str(sample.is_pass()).lower()] += 1
        report = {}
        for test_type, value in summary.items():
            pass_rate = summary[test_type]["true"] / (
                summary[test_type]["true"] + summary[test_type]["false"]
            )
            min_pass_rate = min_pass_dict.get(test_type, default_min_pass_dict)

            if "-" in test_type and summary[test_type]["category"] == "robustness":
                multiple_perturbations_min_pass_rate = min_pass_dict.get(
                    "multiple_perturbations", default_min_pass_dict
                )
                min_pass_rate = min_pass_dict.get(
                    test_type, multiple_perturbations_min_pass_rate
                )
            if summary[test_type]["category"] in ["Accuracy", "performance"]:
                min_pass_rate = 1

            report[test_type] = {
                "category": summary[test_type]["category"],
                "fail_count": summary[test_type]["false"],
                "pass_count": summary[test_type]["true"],
                "pass_rate": pass_rate,
                "minimum_pass_rate": min_pass_rate,
                "pass": pass_rate >= min_pass_rate,
            }

        df_report = pd.DataFrame.from_dict(report, orient="index")
        df_report = df_report.reset_index().rename(columns={"index": "test_type"})

        df_report["pass_rate"] = df_report["pass_rate"].apply(
            lambda x: "{:.0f}%".format(x * 100)
        )
        df_report["minimum_pass_rate"] = df_report["minimum_pass_rate"].apply(
            lambda x: "{:.0f}%".format(x * 100)
        )
        col_to_move = "category"
        first_column = df_report.pop("category")
        df_report.insert(0, col_to_move, first_column)
        df_report = df_report.reset_index(drop=True)

        df_report = df_report.fillna("-")

        return df_report


def multi_model_report(
    summary, min_pass_dict, default_min_pass_dict, generated_results, model_name
):
    for sample in generated_results[model_name]:
        summary[sample.test_type]["category"] = sample.category
        summary[sample.test_type][str(sample.is_pass()).lower()] += 1
    report = {}
    for test_type, value in summary.items():
        pass_rate = summary[test_type]["true"] / (
            summary[test_type]["true"] + summary[test_type]["false"]
        )
        min_pass_rate = min_pass_dict.get(test_type, default_min_pass_dict)

        if summary[test_type]["category"] in ["Accuracy", "performance"]:
            min_pass_rate = 1

        report[test_type] = {
            "model_name": model_name,
            "pass_rate": pass_rate,
            "minimum_pass_rate": min_pass_rate,
            "pass": pass_rate >= min_pass_rate,
        }

    df_report = pd.DataFrame.from_dict(report, orient="index")
    df_report = df_report.reset_index().rename(columns={"index": "test_type"})

    df_report["pass_rate"] = df_report["pass_rate"].apply(
        lambda x: "{:.0f}%".format(x * 100)
    )
    df_report["minimum_pass_rate"] = df_report["minimum_pass_rate"].apply(
        lambda x: "{:.0f}%".format(x * 100)
    )

    df_report = df_report.reset_index(drop=True)
    df_report = df_report.fillna("-")
    return df_report


def color_cells(series, df_final_report):
    res = []
    for x in series.index:
        res.append(
            df_final_report[
                (df_final_report["test_type"] == series.name)
                & (df_final_report["model_name"] == x)
            ]["pass"].all()
        )
    return ["background-color: green" if x else "background-color: red" for x in res]


def mlflow_report(experiment_name, task, df_report, multi_model_comparison=False):
    try:
        import mlflow
    except ModuleNotFoundError:
        print("mlflow package not found. Install mlflow first")

    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # The experiment does not exist, create it
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        # The experiment exists, get its ID
        experiment_id = experiment.experiment_id

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mlflow.start_run(
        run_name=task + "_testing_" + current_datetime,
        experiment_id=experiment_id,
    )

    metrics_to_log = {
        "_pass_rate": lambda row: float(row["pass_rate"].rstrip("%")) / 100,
        "_min_pass_rate": lambda row: float(row["minimum_pass_rate"].rstrip("%")) / 100,
        "_pass_status": lambda row: 1 if row["pass"] else 0,
    }

    if not multi_model_comparison:
        metrics_to_log["_pass_count"] = lambda row: row["pass_count"]
        metrics_to_log["_fail_count"] = lambda row: row["fail_count"]

    for suffix, func in metrics_to_log.items():
        df_report.apply(
            lambda row: mlflow.log_metric(row["test_type"] + suffix, func(row)), axis=1
        )

    mlflow.end_run()


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
