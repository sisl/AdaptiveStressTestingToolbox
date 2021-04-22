============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/sisl/AdaptiveStressTestingToolbox/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

AdaptiveStressTestingToolbox could always use more documentation, whether as part of the
official AdaptiveStressTestingToolbox docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/sisl/AdaptiveStressTestingToolbox/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `AdaptiveStressTestingToolbox` for local development:

1. Fork `AdaptiveStressTestingToolbox <https://github.com/sisl/AdaptiveStressTestingToolbox>`_
   (look for the "Fork" button).

2. Clone your fork locally::

    git clone git@https://github.com:sisl/AdaptiveStressTestingToolbox.git

3. Follow the Git Installation

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes run all the checks and docs builder with `tox <https://tox.readthedocs.io/en/latest/install.html>`_. See the Testing and Documenting sections for more details.

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Testing
-------

We use Travis+Tox to test the Toolbox, and your PR will not be approved if the tests fail, or if the code coverage would drop too low. To avoid this, use tox to test on your local machine.

First, make sure you have all the testig dependencies::

   pip install -r ci/requirements.txt

From the main folder, you can run all tests with verbose output with the following command::

   tox -v

If you only want to check one of the Tox test environments, you can specify which one to run::

   tox -v -e [environment_name]

There are 5 tox environments that are run during the full test:

1. **clean** - Cleans unneeded files from previous tests/development to prepare for testing.
2. **check** - Enforces code formatting. Checks are run using the check-manifest, flake8, and isort packages. You can run the check-autofix tox environment beforehand to fix most issues.
3. **docs** - Builds and checks the documentation.
4. **py36-cover** - Runs the code tests using pytest and codecov.
5. **report** - Reports the code coverage of the previous tests.

Documentation
-------------

The primary form of documentation for the Toolbox is `numpy-style docstrings <https://numpy.org/doc/stable/docs/howto_document.html>`_  within the code. We use these to automatically generate online documentation. If you are changing or adding files, make sure the docstrings are up-to-date.

First, make sure you have the documentation dependencies::

   pip install -r docs/requirements.txt

Some docstring guidelines:
   * Make the descriptions as explanatory as possible.
   * If the parameter has a default value, indicate this by adding "optional" to the type
   * If the type of a parameter is a non-python class (for example, a class from Garage or from elsewhere in the Toolbox), make the type link to that class's documentation. You can do this using `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_.

   For example, to link to a garage class, we first added::

      'garage': ('https://garage.readthedocs.io/en/v2019.10.1/', None)

   to the `intersphinx_mapping` settings in `docs/source/conf.py`. We can then link to a class with the following syntax::

      :domain:'[text to show] <intersphinx_mapping:location>'

   For example, for the garage.experiment.LocalRunner class, we would link using::

      :py:class:`garage.experiment.LocalRunner <garage:garage.experiment.LocalRunner>`

   Note that some links will use a different `domains <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain>`_ . The correct domains and locations can be a bit tricky to find. I recommend using the `sphobjinv package <https://github.com/bskinn/sphobjinv>`_ . For example, we could have run the following command from the terminal to find the correct link syntax::

      sphobjinv suggest -siu https://garage.readthedocs.io/en/v2019.10.1/objects.inv LocalRunner

Once you have updated all of the docstrings, run the following commands from the `docs` folder to update the documentation source and generate a local HTML version for inspection::

   sphinx-apidoc -o ./source/_apidoc ../src/ast_toolbox -eMf
   make clean
   make html

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run ``tox``) [1]_.
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

.. [1] If you don't have all the necessary python versions available locally you can rely on Travis - it will
       `run the tests <https://travis-ci.org/sisl/AdaptiveStressTestingToolbox/pull_requests>`_ for each change you add in the pull request.

       It will be slower though ...
