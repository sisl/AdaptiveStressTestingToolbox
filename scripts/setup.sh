#source ~/miniconda2/bin/activate AST
#cd ~/Research/AdaptiveStressTestingToolbox
unset PYTHONPATH
#export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/garage:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/garage/src
export PYTHONPATH=$PYTHONPATH:$PWD/tests
export PYTHONPATH=$PYTHONPATH:$PWD/src
export PYTHONPATH=$PYTHONPATH:$PWD/examples
#pydot from https://github.com/nlhepler/pydot
#neet run python setup.py install
export PYTHONHASHSEED=0
#cd Toolbox/scripts
