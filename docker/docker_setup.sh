#!/bin/bash -l
set -euo pipefail
apt-get -y update
# Install software-properties-common to be able to add the python repo
apt-get install -y software-properties-common
# Add python repo
add-apt-repository ppa:deadsnakes/ppa
apt-get update
# Install python
apt-get install -y python3.6 python3.6-dev build-essential
apt-get install -y python3.6-venv
# Install packages for bsddb3
apt install -y libdb-dev python3-bsddb3
# Install and update pip
apt install -y python3-pip
pip3 install --upgrade pip
