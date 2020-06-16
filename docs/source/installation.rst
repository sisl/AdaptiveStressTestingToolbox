============
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
