# import setuptools
#
# # Required dependencies
# REQUIRED = [
#             # 'garage==2019.10.1',
#             'numpy==1.14.5',
#             'gym',
#             'python-dateutil',
#             'dowel',
#             'joblib',
#             'cached_property',
#             'akro',
#             'glfw',
#             'pyprind',
#             'cma',
#             'bsddb3',
#             'fire',
#             'depq',
#             'compress_pickle',
#             'pydot',]
#
# # Dependencies for optional features
# EXTRAS = {}
#
# with open("README.md", "r") as fh:
#     long_description = fh.read()
#
# setuptools.setup(
#     name="ast-toolbox",
#     version="2020.06.01.dev1",
#     author="Stanford Intelligent Systems Laboratory",
#     author_email="mkoren@stanford.edu",
#     description="A toolbox for worst-case validation of autonomous policies",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/sisl/AdaptiveStressTestingToolbox",
#     # packages=setuptools.find_packages(where='./ast_toolbox'),
#     packages=setuptools.find_packages(),
#     classifiers=[
#         "License :: OSI Approved :: MIT License",
#
#         "Operating System :: OS Independent",
#
#         'Intended Audience :: Developers',
#         'Intended Audience :: Education',
#         'Intended Audience :: Science/Research',
#         'Topic :: Software Development :: Libraries',
#         'Topic :: Software Development :: Testing',
#         'Topic :: Scientific/Engineering :: Mathematics',
#         'Topic :: Scientific/Engineering :: Artificial Intelligence',
#
#         "Programming Language :: Python :: 3.6",
#         "Programming Language :: Python :: 3.7",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3.9",
#
#         'Development Status :: 3 - Alpha',
#     ],
#     python_requires='>=3.6',
#     project_urls={
#         'Source': 'https://github.com/sisl/AdaptiveStressTestingToolbox',
#         'Documentation':'https://ast-toolbox.readthedocs.io/en/master/',
#         'Tracker':'https://github.com/sisl/AdaptiveStressTestingToolbox/issues',
#         'Status':'https://travis-ci.org/github/sisl/AdaptiveStressTestingToolbox',
#         'Testing':'https://codecov.io/gh/sisl/AdaptiveStressTestingToolbox',
#     },
#     install_requires=REQUIRED,
#     extras_require=EXTRAS,
# )
#

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='ast-toolbox',
    version='2020.06.01.dev1',
    license='MIT',
    description='A toolbox for worst-case validation of autonomous policies',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Stanford Intelligent Systems Laboratory',
    author_email='mkoren@stanford.edu',
    url='https://github.com/sisl/AdaptiveStressTestingToolbox',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
        'Private :: Do Not Upload',
    ],
    project_urls={
        'Documentation': 'https://AdaptiveStressTestingToolbox.readthedocs.io/',
        'Changelog': 'https://AdaptiveStressTestingToolbox.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/sisl/AdaptiveStressTestingToolbox/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={
        'console_scripts': [
            'ast-toolbox = ast_toolbox.cli:main',
        ]
    },
)
