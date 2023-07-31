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
This guides us to the local mlflow tracking server. It has model-name specified as `experiment-name` and task-date specified as `run-name`. To check the logs, select the run-name and go to the metrics section. Mlflow helps us to maintain and compare different test configurations of same or different model runs. We can compare the metrics of different runs

![MLFlow Tracking Server](https://github.com/JohnSnowLabs/langtest/blob/chore/webiste_updates/docs/assets/images/mlflow/experiment_run_name.png?raw=true)


</div><div class="h3-box" markdown="1">

</div></div>