#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# pyConnectomist current version
version_major = 1
version_minor = 0
version_micro = 2

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Python processing for PreClinical data.
"""
SUMMARY = """
.. container:: summary-carousel

    pypreclin is a Python module for **processing preclinical data** (non-human
    primate) that offers:

    * pypreclin_preproc_fmri: a script to pre-process fMRI data.
    * pypreclin_prepare_target: a script to perform B0 ihomogeneities
      correction using a non linear registration to the anatomical image.
    * pypreclin_preproc_multi_fmri: a script to pre-process BIDS fMRI data.
"""
long_description = (
    "pypreclin is a Python project that provides a collection of Python "
    "scripts for processing MRI preclinical datasets. This work is made "
    "available by a community of people, amoung which the CEA Neurospin "
    "UNATI and CEA NeuroSpin UNICOG laboratories, in particular A. Grigis, "
    "J. Tasserie, and B. Jarraya.\n")

# Main setup parameters
NAME = "pypreclin"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/neurospin/pypreclin"
DOWNLOAD_URL = "https://github.com/neurospin/pypreclin"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
Antoine Grigis <antoine.grigis@cea.fr>
Jordy Tasserie <jordy.tasserie@cea.fr>
Bechir Jarraya <bechir.jarraya@cea.fr>
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["pypreclin"]
REQUIRES = [
    "dipy>=1.11.0",
    "filelock>=3.0.12",
    "hopla==1.0.5",
    "joblib>=0.13.2",
    "matplotlib>=3.8.0",
    "networkx>=3.1",
    "nilearn>=0.11.1",
    "nibabel>=1.1.0",
    "nipype>=1.0.1",
    "numpy>=1.11.0",
    "progressbar2>=4.5.0",
    "setuptools==66",
    "wheel==0.38.4",
    "pyconnectome>=1.0.0",
    "pyconnectomist>=2.0.0",
    "pydcmio>=2.0.2",
    "pyfreesurfer>=1.2.0",
    "scipy>=0.17.0",
    "transforms3d>=0.3.1",
    "torch>=2.7.0",
    "torchvision>=0.22.0"
]
EXTRA_REQUIRES = {
    "gui": {
        "python-pypipe>=0.0.1"
    }
}
EXTRANAME = "UNICOG"
EXTRAURL = "http://www.unicog.org/site_2016/"
