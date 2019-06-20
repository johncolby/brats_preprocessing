from setuptools import setup

setup(
    name='brats_preprocessing',
    version='0.1.0',
    url='https://github.com/johncolby/brats_preprocessing',
    author='John Colby',
    author_email='john.b.colby@gmail.com',
    description='Tools for preprocessing clinical data according to BraTS specification',
    packages=['brats_preprocessing'],
    install_requires=['nibabel', 'nipype', 'pandas', 'requests', 'pydicom'],
    extras_require={'classify_series': ['rpy2', 'tzlocal'], 'report': ['rpy2', 'tzlocal'], 'download': ['air_download']},
    include_package_data=True,
    entry_points={'console_scripts': ['process_gbm = brats_preprocessing.brats_preprocessing:cli']},
)