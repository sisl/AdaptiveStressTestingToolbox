#!/bin/bash
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    brew install berkeley-db
    export BERKELEYDB_DIR=$(brew --cellar)/berkeley-db/5.3
    export YES_I_HAVE_THE_RIGHT_TO_USE_THIS_BERKELEY_DB_VERSION=1
else
    sudo apt-get update && sudo apt install libdb-dev python3-bsddb3 graphviz
fi
python -mpip install --progress-bar=off tox -rci/requirements.txt
virtualenv --version
easy_install --version
pip --version
tox --version
