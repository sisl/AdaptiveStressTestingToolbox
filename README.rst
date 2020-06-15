===============================
Adaptive Stress Testing Toolbox
===============================
|build-status| |docs| |coverage| |license|

========
Overview
========

A toolbox for worst-case validation of autonomous policies

Adaptive Stress Testing is a worst-case validation method for autonomous policies. This toolbox is currently under construction, and is being actively developed by the Stanford Intelligent Systems Lab.

See https://ast-toolbox.readthedocs.io/en/master/ for documentation.

Maintained by the Stanford Autonomous Systems Lab


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


.. |build-status| image:: https://travis-ci.org/sisl/AdaptiveStressTestingToolbox.svg
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/sisl/AdaptiveStressTestingToolbox

.. |docs| image:: https://readthedocs.org/projects/ast-toolbox/badge/?version=master
    :alt: build status
    :scale: 100%
    :target: https://ast-toolbox.readthedocs.io/en/master/?badge=master

.. |coverage| image:: https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox/branch/master/graph/badge.svg
    :alt: build status
    :scale: 100%
    :target: https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox

.. |license| image:: https://img.shields.io/badge/license-MIT-yellow.svg
    :alt: build status
    :scale: 100%
    :target: https://github.com/sisl/AdaptiveStressTestingToolbox/blob/master/LICENSE
