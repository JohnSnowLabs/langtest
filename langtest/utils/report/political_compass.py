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
