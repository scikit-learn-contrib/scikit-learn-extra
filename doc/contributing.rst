..
    Contribution code partially copied from https://github.com/scikit-learn-contrib/category_encoders

Contributing
============

We welcome and in fact would love some help.

How to Contribute
^^^^^^^^^^^^^^^^^

The preferred workflow to contribute is:

1. Fork this repository into your own github account.
2. Clone the fork on your account onto your local disk:

.. code-block:: console

    git clone git@github.com:YourLogin/scikit-learn-extra.git
    cd scikit-learn-extra

3. Create a branch for your new feature, do not work in the master branch:

.. code-block:: console

    git checkout -b new-feature

4. Write some code, or docs, or tests.
5. When you are done, submit a pull request.

Guidelines
^^^^^^^^^^

This is still a very young project, but we do have a few guiding principles:

1. Maintain semantics of the scikit-learn API
2. Write detailed docstrings in numpy format
3. Support pandas dataframes and numpy arrays as inputs
4. Write tests
5. Format with black. You can use `pre-commit <https://pre-commit.com/>`_ to auto-format code on each commit,

   .. code-block:: console

       pip install pre-commit
       pre-commit install

Running Tests
^^^^^^^^^^^^^

To run the tests, use:

.. code-block:: console

    pytest

Easy Issues / Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are usually some issues in the project github page looking for contributors, if not you're welcome to propose some
ideas there, or a great first step is often to just use the library, and add to the examples directory. This helps us
with documentation, and often helps to find things that would make the library better to use.
