---
layout: docs
seotitle: MLFlow Tracking | LangTest | John Snow Labs
title: MLFlow Tracking
permalink: /docs/pages/docs/ml_flow
key: docs-install
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

For tracking the run metrics (logs) on mlflow, you just need to define a flag: `mlflow_tracking = True` in the report method and your metrics will be logged on local mlflow tracking server.

```python
h.report(mlflow_tracking = True)
!mlflow ui
```
When you set `mlflow_tracking = True` in your report method, it initiates the tracking feature of MLflow. This leads us to a locally hosted MLflow tracking server.

On this server, each run of your model is represented as an experiment. These experiments are identified by a unique experiment-name, which corresponds to the model-name. Alongside this, each experiment run is time-stamped and labeled as a run-name that corresponds to the task-date.

If you want to review the metrics and logs of a specific run, you simply select the associated run-name. This will guide you to the metrics section, where all logged details for that run are stored. This system provides an organized and streamlined way to keep track of each model's performance during its different runs.

The tracking server looks like this with experiments and run-names specified in following manner:

![MLFlow Tracking Server](/assets/images/mlflow/experiment_run_name.png)

To check the metrics, select the run-name and go to the metrics section.

![MLFlow Metrics Checking](/assets/images/mlflow/checking_metrics.png)

If you decide to run the same model again, whether with the same or different test configurations, MLflow will log this as a distinct entry in its tracking system.

Each of these entries captures the specific state of your model at the time of the run, including the chosen parameters, the model's performance metrics, and more. This means that for every run, you get a comprehensive snapshot of your model's behavior under those particular conditions.

You can then use the compare section to get a detailed comparison for the different runs.

![MLFlow Run Comparisons](/assets/images/mlflow/compare_runs.png)

![MLFlow Run Comparisons Detailed](/assets/images/mlflow/view_comparisons.png)

Thus, MLflow acts as your tracking system, recording the details of each run, and providing a historical context to the evolution and performance of your model. This capability is instrumental in maintaining a disciplined and data-driven approach to improving machine learning models.




</div><div class="h3-box" markdown="1">

</div></div>