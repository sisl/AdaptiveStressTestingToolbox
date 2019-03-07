Installation Guide
******************

Open the command prompt and change to the directory you wish to download the package into. Run the following command to clone the repository, as well as the submodules, and change to the top-level directory:
::
	git clone --recursive https://github.com/mark-koren/DRL-AST.git
	cd <Path-To-DRL-AST>/DRL-AST

If you have already cloned the repo, you can run the following command to download the submodules:
::
	cd <Path-To-DRL-AST>/DRL-AST
	git submodule update --init --recursive

Create the Anaconda environment by running the following command from the top-level directory:
::
	conda env create -f environment.yml

Once the environment has been created, activate it by running:
::
	source activate DRL-AST

More information on Anaconda environments can be found in their documentation. Finally, add rllab to your ``PYTHONPATH`` like shown below:
::
	export PYTHONPATH=$PYTHONPATH:<Path-To-Install>/DRL-AST/rllab

To validate your installation, please run the following:
::
	python runners/validate_install.py

You should see the method run for a single iteration, then print out a success message. You are now able to run this package. For more information on how to interface this package with your work, please see the tutorial.
