#source ~/miniconda2/bin/activate AST
#cd ~/Research/AdaptiveStressTestingToolbox
unset PYTHONPATH
#export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/garage:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD/Toolbox/garage/src
export PYTHONPATH=$PYTHONPATH:$PWD/TestCases
export PYTHONPATH=$PYTHONPATH:$PWD/Toolbox
#pydot from https://github.com/nlhepler/pydot
#neet run python setup.py install
export PYTHONPATH=$PYTHONPATH:$PWD/Toolbox/pydot
export PYTHONHASHSEED=0
#cd Toolbox/scripts
