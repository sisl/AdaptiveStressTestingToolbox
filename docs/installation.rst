Installation Guide
******************

Open the command prompt and change to the directory you wish to download the package into. Run the following command to clone the repository, as well as the submodules, and change to the top-level directory:
::
	git clone --recursive https://github.com/sisl/AdaptiveStressTestingToolbox
	cd <Path-To-AdaptiveStressTestingToolbox>/AdaptiveStressTestingToolbox

If you have already cloned the repo, you can run the following command to download the submodules:
::
	cd <Path-To-AdaptiveStressTestingToolbox>/AdaptiveStressTestingToolbox
	git submodule update --init --recursive

If you have done the previous steps correctly, there should be a ``AdaptiveStressTestingToolbox/garage`` folder. Please see the `garage installation page <https://rlgarage.readthedocs.io/en/latest/user/installation.html>`_ for details on how to get all of their dependencies. Once garage is installed, create the Conda environment by running the following command from the top-level garage directory:
::
	cd <Path-To-AdaptiveStressTestingToolbox>/AdaptiveStressTestingToolbox/Toolbox/garage
	conda create --name <your_environment_name> python=3.6

Once the environment has been created, activate it by running:
::
	conda activate <your_environment_name>

More information on Conda environments can be found in their `documentation <https://conda.io/en/latest/>`_. Next, run the following commands to install all of the dependencies. Note, you may run into errors with installing dm-control. You can ingore these if you are not planning to use Mujoco. 
::
	cd Toolbox/garage
	touch mjkey.txt
	echo "hello" > mjkey.txt
	./scripts/setup_linux.sh --mjkey mjkey.txt --modify-bashrc
	rm mjkey.txt
	pip install --ignore-installed garage
	pip install -e .[dev]
	cd ../../

Finally, add everything to your ``PYTHONPATH`` like shown below:
::
	source setup.sh

To validate your installation, please run the following:
::
	python <Path-To-AdaptiveStressTestingToolbox>/AdaptiveStressTestingToolbox/TestCases/validate_install.py

You should see the method run for a single iteration, then print out a success message. You are now able to run this package. For more information on how to interface this package with your work, please see the tutorial.
