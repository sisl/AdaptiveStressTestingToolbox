#!/bin/bash
python --version
uname -a
lsb_release -a || true
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    [[ $TOXENV =~ py3 ]] && brew upgrade python
    [[ $TOXENV =~ py2 ]] && brew install python@2
    export PATH="/usr/local/opt/python/libexec/bin:${PATH}"
fi
