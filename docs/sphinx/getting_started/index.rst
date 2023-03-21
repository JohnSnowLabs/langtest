###############
Quick Start
###############

*********************
NLP Test Quick Start
*********************

The following can be used as a quick reference on how to get up and running with ``nlptest``:

.. code-block:: bash

    # Install Spark NLP from PyPI
    pip install nlptest==|release|

.. code-block:: python

    from nlptest import Harness

    # Create a Harness object
    h = Harness('ner', model='dslim/bert-base-NER', hub='transformers')

    # Generate, run and get a report on your test cases
    h.generate().run().report()

Installation
================

We can create a Python `Virtualenv <https://virtualenv.pypa.io/en/latest/>`_:

.. code-block:: bash

    virtualenv nlptest --python=python3.8
    source nlptest/bin/activate
    pip install nlptest==|release| jupyter

Now you should be ready to create a jupyter notebook with NLP Test running:

.. code-block:: bash

    jupyter notebook

We can also use conda and create a new `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment to manage all the dependencies there.

Then we can create a new environment ``nlptest`` and install the ``nlptest`` package with pip:

.. code-block:: bash

    conda create -n nlptest python=3.8 -y
    conda activate nlptest
    conda install -c nlptest==|release| jupyter

Now you should be ready to create a jupyter notebook with NLP Test running:

.. code-block:: bash

    jupyter notebook

