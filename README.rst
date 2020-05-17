# AST
[![Build Status](https://travis-ci.org/sisl/AdaptiveStressTestingToolbox.svg?branch=master)](https://travis-ci.org/sisl/AdaptiveStressTestingToolbox)
[![Documentation Status](https://readthedocs.org/projects/ast-toolbox/badge/?version=master)](https://ast-toolbox.readthedocs.io/en/master/?badge=master)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/sisl/AdaptiveStressTestingToolbox/blob/master/LICENSE)
[![codecov](https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox)

Adaptive Stress Testing is a worst-case validation method for autonomous policies. This toolbox is currently under construction, and is being actively developed by the Stanford Intelligent Systems Lab.

See https://ast-toolbox.readthedocs.io/en/master/ for documentation.

========
Overview
========

A toolbox for worst-case validation of autonomous policies

* Free software: MIT license

Installation
============

::

    pip install ast-toolbox

You can also install the in-development version with::

    pip install git+ssh://git@https://github.com/sisl/AdaptiveStressTestingToolbox.git@master

Documentation
=============


https://ast-toolbox.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
