import pandas as pd


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
