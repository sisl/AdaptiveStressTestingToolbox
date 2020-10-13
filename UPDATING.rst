================
UPDATING VERSION
================

When updating version with bumbpversion, all version locations will be updated.

A tagged commit will also be pushed to github.

Settings for bumpversion are in .bumpversion.cfg

To set a new version date (for example, 2020.09.01) use the following command:

bumpversion --new-version 2020.09.01.dev1 release --verbose --list --allow-dirty

To change release type (for example, 2020.09.01.dev1 to 2020.09.01.rc0), use:

bumpversion release --verbose --list --allow-dirty

To change release number (for example, 2020.09.01.dev0 to 2020.09.01.dev1), use:

bumpversion relnumber -n --verbose --list --allow-dirty

=============
UPDATING PYPI
=============
TRAVIS DOES THIS AUTOMATICALLY - YOU SHOULD NOT NEED TO DO THIS

Make sure you have the dependencies installed:

python3 -m pip install --user --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine


First, delete the /dist directory, if it exists. Then run the following command
to generate a build in the /dist directory:

python3 setup.py sdist bdist_wheel

Now upload with the following command:

python3 -m twine upload dist/*
