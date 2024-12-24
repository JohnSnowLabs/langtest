---
layout: docs
seotitle: Integrations | LangTest | John Snow Labs
title: Integrations
permalink: /docs/pages/docs/integrations
key: docs-integrations
modify_date: "2023-03-28"
header: true
---

<div class="main-docs" markdown="1">
<div class="h3-box" markdown="1">


**LangTest** is an open-source Python library that empowers developers to build safe and reliable Natural Language Processing (NLP) models. It seamlessly integrates with popular platforms and tools, including **Databricks**, enabling scalable testing and evaluation. Install LangTest easily using pip to enhance your NLP workflows.

</div>
<div class="h3-box" markdown="1">

## Databricks

**Introduction**  
LangTest is a powerful tool for testing and evaluating NLP models, and integrating it with Databricks allows users to scale their testing with large datasets and leverage real-time analytics. This integration streamlines the process of assessing model performance, ensuring high-quality results while maintaining scalability and efficiency. With Databricks, LangTest becomes an even more versatile solution for NLP practitioners working with substantial data pipelines and diverse datasets.  

**Prerequisites**  
Before starting, ensure you meet the following requirements. You need access to a Databricks Workspace and an installed version of the `LangTest` package (version `2.5.0` or `later`). Additionally, make sure you have your Databricks API keys or credentials ready and have Python (version 3.9 or later) installed on your system. Optionally, access to sample datasets is helpful for testing and exploring features during your initial setup.  

#### **Step-by-Step Setup**

Getting started with LangTest and Databricks is straightforward and involves a few simple steps. Follow the instructions below to set up and run your first NLP model test.

1. **Install LangTest and Dependencies**  
   Begin by installing LangTest using pip:
   ```bash
   pip install langtest==2.5.0
   ```
   Ensure all required dependencies are installed and your environment is ready.

2. **Load Datasets from Databricks**  
   Use the Databricks connector to load data directly into your LangTest pipeline:
   ```python
   from pyspark.sql import DataFrame

    # Load the dataset into a Spark DataFrame
    df: DataFrame = spark.read.json("<FILE_PATH>")

   ```
   print the dataframe schema
   ```python
   df.printSchema()
    ```

3. **Configuration**  
   In this section, we will configure the tests, datasets, and model settings required to effectively use LangTest. This includes setting up the test parameters, loading datasets, and defining the model configuration to ensure seamless integration and accurate evaluation.

   - **Tests Config:**

    ```python
    test_config = {
        "tests": {
            "defaults": {"min_pass_rate": 1.0},
            "robustness": {
                "add_typo": {"min_pass_rate": 0.7},
                "lowercase": {"min_pass_rate": 0.7},
            },
        },
    }
    ```

   - **Dataset Config:**

    ```python
    input_data = {
        "data_source": df,
        "source": "spark",
        "spark_session": spark # make sure that spark session is started or not
    }
    ```

   - **Model Config:**

    ```python
    model_config = {
        "model": {
            "endpoint": "databricks-meta-llama-3-1-70b-instruct",
        },
        "hub": "databricks",
        "type": "chat"
    }
    ```


4. **Set Up and Run Tests with Harness**  
   Use the `Harness` class to configure, generate, and execute tests. Define your task, model, data, and configuration:

   ```python
    harness = Harness(
        task="question-answering",
        model=model_config,
        data=input_data,
        config=test_config
    )
   ```

   Generate and Execute the testcases on model to evaluate with langtest:
   ```python
   harness.generate().run().report()
   ```

   To Review the Testcases:
   ```python
   testcases_df = harness.testcases()
   testcases_df
   ```

   To save testcases in delta live tables
   ```python
   import os
   from deltalake import DeltaTable
   from deltalake.writer import write_deltalake

    write_deltalake("tmp/langtest_testcases", testcases_df) # for existed tables, pass mode="append"

   ```

   To Review the Generated Results 
   ```python
   results_df = harness.generated_results()
   results_df
   ```

   Similary, for results_df in delta live tables.
   ```python
   import os
   from deltalake import DeltaTable
   from deltalake.writer import write_deltalake

    write_deltalake("tmp/langtest_generated_results", results_df) # for existed tables, pass mode="append"

   ```

   This process evaluates your model's performance on the loaded data and provides a comprehensive report of the results.

By following these steps, you can easily integrate Databricks with LangTest to perform NLP or LLM model testing. If you encounter issues during setup or execution, refer to the troubleshooting section for solutions.

**Troubleshooting & Support**  
While setting up, you may encounter common issues like authentication errors with Databricks, incorrect dataset paths, or model compatibility problems. To resolve these, verify your API keys and workspace URL, ensure the specified dataset exists in Databricks, and confirm that your LangTest version is compatible with your project. If further help is needed, explore the FAQ section, access detailed documentation, or reach out through the support channels or community forum for assistance.

### FAQ

**Q: How do I resolve authentication errors with Databricks?**  
A: Ensure that your API keys and workspace URL are correct. Double-check that your credentials have the necessary permissions to access the Databricks workspace.

**Q: What should I do if the dataset path is incorrect?**  
A: Verify that the specified dataset exists in Databricks and that the path is correctly formatted. You can use the Databricks UI to navigate and confirm the dataset location.

**Q: How can I check if my LangTest version is compatible with my project?**  
A: Refer to the LangTest documentation for version compatibility information. Ensure that you are using a version of LangTest that supports the features and integrations required for your project.

**Q: Where can I find more detailed documentation?**  
A: Access the detailed documentation on the LangTest official website or the Databricks documentation portal for comprehensive guides and examples.

**Q: How can I get additional support?**  
A: Reach out through the support channels provided by LangTest or Databricks. You can also join the community forum to ask questions and share experiences with other users.


</div></div>