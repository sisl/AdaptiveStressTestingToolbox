===============================
Adaptive Stress Testing Toolbox
===============================
v2020.09.01.1.

|build-status| |docs| |coverage| |license|

========
Overview
========

A toolbox for worst-case validation of autonomous policies.

Adaptive Stress Testing is a worst-case validation method for autonomous policies. This toolbox is being actively developed by the Stanford Intelligent Systems Lab.

See https://ast-toolbox.readthedocs.io/en/latest/ for documentation.

Maintained by the `Stanford Intelligent Systems Lab (SISL) <http://sisl.stanford.edu/>`_


* Free software: MIT license

Installation
============

Pip Installation Method
-----------------------

You can install the latest stable release from pypi::

    pip install ast-toolbox

You can also install the latest version with::

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

Git Installation Method
-----------------------
If you are interested in development, you should clone the repo. You can use https::

   git clone https://github.com/sisl/AdaptiveStressTestingToolbox.git

You can also use ssh::

   git clone git@github.com:sisl/AdaptiveStressTestingToolbox.git

If you are on Linux, use the following commands to setup the Toolbox::

   cd AdaptiveStressTestingToolbox
   git submodule update --init --recursive
   sudo chmod a+x scripts/install_all.sh
   sudo scripts/install_all.sh
   source scripts/setup.sh

Documentation
=============


You can find our `documentation <https://ast-toolbox.readthedocs.io/en/latest/>`_ on readthedocs.


Development
===========

Please see our `Contributions Guide <https://ast-toolbox.readthedocs.io/en/latest/contributing.html>`_.

Acknowledgements
================

Built using the `cookiecutter-pylibrary <https://github.com/ionelmc/cookiecutter-pylibrary>`_ by Ionel Cristian Mărieș


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
    :target: https://app.codecov.io/gh/sisl/AdaptiveStressTestingToolbox

.. |license| image:: https://img.shields.io/badge/license-MIT-yellow.svg
    :alt: License
    :scale: 100%
    :target: https://github.com/sisl/AdaptiveStressTestingToolbox/blob/master/LICENSE
