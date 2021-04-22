# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('./'))
# sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../src/ast_toolbox/'))
# sys.path.insert(0, os.path.abspath('../src/'))
# sys.path.insert(0, os.path.abspath('../../src/ast_toolbox/'))
sys.path.insert(0, os.path.abspath('../../src/'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
# sphinx.ext.autodoc config
autodoc_mock_imports = ['bsddb3']
autodoc_member_order = 'groupwise'

source_suffix = '.rst'
master_doc = 'index'
project = 'AdaptiveStressTestingToolbox'
year = '2018-2020'
author = 'Stanford Intelligent Systems Laboratory'
copyright = '{0}, {1}'.format(year, author)
version = release = '2020.09.01.2'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://https://github.com/sisl/mark-koren/AdaptiveStressTestingToolbox/issues/%s', '#'),
    'pr': ('https://https://github.com/sisl/mark-koren/AdaptiveStressTestingToolbox/pull/%s', 'PR #'),
}
linkcheck_retries = 3

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

# sphinx.ext.intersphinx config
intersphinx_mapping = {
    'garage_latest': ('https://garage.readthedocs.io/en/latest/', None),
    'garage': ('https://garage.readthedocs.io/en/v2019.10.1/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# sphinx.ext.napoleon config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
