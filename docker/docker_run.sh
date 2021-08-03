#!/bin/bash -l
set -euo pipefail
exec pyperformance run -f -b 2to3,unpickle_pure_python,django_template -o "$1"
