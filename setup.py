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


REQUIRED = ['garage==2019.10.1',
            # 'numpy>=1.14.5',
            'gym==0.12.4',
            'python-dateutil',
            'dowel',
            'joblib',
            'cached_property',
            'akro',
            'glfw',
            'pyprind',
            'cma',
            'fire',
            'depq',
            'compress_pickle',
            'pydot',
            ]

# Dependencies for optional features
EXTRAS = {}

EXTRAS['all'] = list(set(sum(EXTRAS.values(), [])))

EXTRAS['ge'] = ['bsddb3']

setup(
    name='ast-toolbox',
    version='2020.09.01.2',
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
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={
        'console_scripts': [
            'ast-toolbox = ast_toolbox.cli:main',
        ]
    },
)
