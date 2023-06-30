###############
Quick Start
###############

LangTest Quick Start
=======================

The following can be used as a quick reference on how to get up and running with ``langtest``:

.. code-block:: bash
    :substitutions:

    # Install langtest from PyPI
    pip install langtest==|release|


.. code-block:: python

    from langtest import Harness

    # Create a Harness object
    h = Harness('ner', model='dslim/bert-base-NER', hub='transformers')

    # Generate, run and get a report on your test cases
    h.generate().run().report()

Alternative Installation Options
================================

We can create a Python `Virtualenv <https://virtualenv.pypa.io/en/latest/>`_:

.. code-block:: bash
    :substitutions:

    virtualenv langtest --python=python3.8
    source langtest/bin/activate
    pip install langtest==|release| jupyter

Now you should be ready to create a jupyter notebook with LangTest running:

.. code-block:: bash

    jupyter notebook

We can also use conda and create a new `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ environment to manage all the dependencies there.

Then we can create a new environment ``langtest`` and install the ``langtest`` package with pip:

.. code-block:: bash
    :substitutions:

    conda create -n langtest python=3.8 -y
    conda activate langtest
    conda install -c langtest==|release| jupyter

Now you should be ready to create a jupyter notebook with LangTest running:

.. code-block:: bash

    jupyter notebook

