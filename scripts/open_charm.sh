source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate AST
cd ..
unset PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/third_party/garage/src:$(pwd)/src:$PYTHONPATH
cd ..
charm AdaptiveStressTestingToolbox
