import os
import setuptools

import fluoressential


def read(*names):
    values = {}
    extensions = ['.txt', '.rst', '.md']
    for name in names:
        value = ''
        for extension in extensions:
            filename = name + extension
            if os.path.isfile(filename):
                value = open(filename).read()
                break
        values[name] = value
    return values


setuptools.setup(
    name=fluoressential.__name__,
    version=fluoressential.__version__,
    description=fluoressential.__description__,
    long_description="""%(README)s""" % read('README'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords=fluoressential.__keywords__,
    author=fluoressential.__author__,
    author_email=fluoressential.__email__,
    maintainer=fluoressential.__author__,
    maintainer_email=fluoressential.__email__,
    url=fluoressential.__website__,
    license=fluoressential.__license__,
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'scikit-image', 'opencv-python',
        'natsort', 'lmfit', 'tqdm', 'scikits.odes'
    ]
)
