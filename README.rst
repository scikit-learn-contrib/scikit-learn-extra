.. -*- mode: rst -*-

|PyPi|_ |Azure|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |PyPi| image:: https://badge.fury.io/py/scikit-learn-extra.svg
.. _PyPi: https://badge.fury.io/py/scikit-learn-extra

.. |Azure| image:: https://dev.azure.com/scikit-learn-extra/scikit-learn-extra/_apis/build/status/scikit-learn-contrib.scikit-learn-extra?branchName=master
.. _Azure: https://dev.azure.com/scikit-learn-extra/scikit-learn-extra/_build/latest?definitionId=1&branchName=master

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/scikit-learn-extra

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/scikit-learn-extra.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/scikit-learn-extra/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/scikit-learn-extra/badge/?version=latest
.. _ReadTheDocs: https://sklearn-template.readthedocs.io/en/latest/?badge=latest

scikit-learn-extra - A set of useful tools compatible with scikit-learn
=======================================================================

.. _scikit-learn: https://scikit-learn.org

scikit-learn-extra is a Python module for machine learning that extends scikit-learn. It includes algorithms that are useful but do not satisfy the scikit-learn `inclusion criteria <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_, for instance due to their novelty or lower citation number.

Installation
------------

Dependencies
^^^^^^^^^^^^

scikit-learn-extra requires,
 
- Python (>=3.5)
- scikit-learn (>=0.21), and its dependencies


User installation
^^^^^^^^^^^^^^^^^

Latest release can be installed with conda,

.. code::

   conda install -c conda-forge scikit-learn-extra

or from PyPi with,

.. code::
   
   pip install scikit-learn-extra

Note that installing from PyPi requires a working C compiler (cf `installation
instructions
<https://scikit-learn.org/dev/developers/advanced_installation.html#platform-specific-instructions>`_).
   
The developement version can be installed with,

.. code::

    pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip


License
-------

This package is released under the 3-Clause BSD license.
