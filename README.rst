===============================
Adaptive Stress Testing Toolbox
===============================
|build-status| |docs| |coverage| |license|

========
Overview
========

A toolbox for worst-case validation of autonomous policies

Adaptive Stress Testing is a worst-case validation method for autonomous policies. This toolbox is being actively developed by the Stanford Intelligent Systems Lab.

See https://ast-toolbox.readthedocs.io/en/latest/ for documentation.

Maintained by the Stanford Autonomous Systems Lab


* Free software: MIT license

Installation
============

At the command line::

    pip install ast-toolbox

You can also install the in-development version with::

    pip install git+ssh://git@https://github.com/sisl/AdaptiveStressTestingToolbox.git@master

Using the Go-Explore work requires having a Berkely DB installation findable on your system. If you are on Linux::

   sudo apt-get update
   sudo apt install libdb-dev python3-bsddb3

If you are on OSX::

   brew install berkeley-db
   export BERKELEYDB_DIR=$(brew --cellar)/berkeley-db/5.3
   export YES_I_HAVE_THE_RIGHT_TO_USE_THIS_BERKELEY_DB_VERSION=1

Once you have the Berkeley DB system dependency met, you can install the toolbox::

   pip install ast-toolbox[ge]

Documentation
=============


https://ast-toolbox.readthedocs.io/en/latest/


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


.. |build-status| image:: https://api.travis-ci.org/sisl/AdaptiveStressTestingToolbox.svg
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.org/sisl/AdaptiveStressTestingToolbox

.. |docs| image:: https://readthedocs.org/projects/ast-toolbox/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://ast-toolbox.readthedocs.io/en/latest/?badge=latest

.. |coverage| image:: https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox/branch/master/graph/badge.svg
    :alt: Code Coverage
    :scale: 100%
    :target: https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox

.. |license| image:: https://img.shields.io/badge/license-MIT-yellow.svg
    :alt: License
    :scale: 100%
    :target: https://github.com/sisl/AdaptiveStressTestingToolbox/blob/master/LICENSE
