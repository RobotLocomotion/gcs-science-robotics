from setuptools import setup, find_packages

setup(
    name='gcs',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'lxml',
        'matplotlib',
        'networkx',
        'notebook',
        'numpy',
        'pandas',
        'pydot',
        'scipy',
    ]
)
