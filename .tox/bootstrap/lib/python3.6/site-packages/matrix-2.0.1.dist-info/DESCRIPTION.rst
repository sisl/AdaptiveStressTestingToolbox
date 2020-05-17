========
Overview
========



Generic matrix generator.

* Free software: BSD license

Installation
============

::

    pip install matrix

Documentation
=============

https://python-matrix.readthedocs.io/

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


Changelog
=========

2.0.0 (2016-02-03)
------------------

* Switch to configparser2 (the Python 3.5 backport)
* Drop Python 2.6 support.

1.3.1 (2015-04-15)
------------------

* Fix the samefile check on windows.

1.3.0 (2015-04-15)
------------------

* Added an optional ``[cli]`` extra (use ``pip install matrix[cli]``) that enables a ``matrix-render`` command.
  The new command loads configuration and passes the generated matrix to the given Jinja2 templates.

1.2.0 (2015-04-03)
------------------

* Fix handling when having aliased entries that have empty ("-") values.

1.1.0 (2015-02-12)
------------------

* Add support for empty inclusions/exclusions.

1.0.0 (2014-08-09)
------------------

* Fix Python 2.6 support.
* Add support for really empty entries (leave completely empty instead of "-")


0.5.0 (2014-08-09)
------------------

* Fix Python 3 support.


