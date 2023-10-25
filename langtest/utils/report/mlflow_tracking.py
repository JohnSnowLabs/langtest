import datetime


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
