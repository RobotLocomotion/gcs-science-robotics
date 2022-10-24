# GCS - Motion Planning around Obstacles with Convex Optimization

This code has been updated since the Arxiv paper was published. The code used to run the examples in that paper can be found under the [`arxiv_paper_version`](https://github.com/mpetersen94/gcs/releases/tag/arxiv_paper_version) tag. That version was tested to work with Drake version 1.3 (although version up to 1.8 should work with some deprecation warnings).

## Running via Deepnote
Most of the examples and reproductions can be run on [Deepnote](https://deepnote.com/workspace/mark-petersen-2785519d-2c3e-430b-9a10-a1754f2de37d/project/GCS-Motion-Planning-around-Obstacles-with-Convex-Optimization-3afac8e3-cbc0-41d1-9afb-0d38dfbe9ffa/).

After duplicating the project into your own account, be sure to run the `MosekLicenseUpload.ipynb` notebook to make your Mosek License available for solving the optimization problems.

Note: The PRM and Bimanual reproductions do not yet work on Deepnote and the UAV and Maze reproductions have been shrunk in size to avoid hitting memory limits on Deepnote.

## Running locally

### Installing Dependencies
This code depends on [Drake](https://drake.mit.edu), specifically its Python bindings. To install Drake bindings follow the instruction on [Drake's Installation Page](https://drake.mit.edu/installation.html). Any of the installation methods listed should work.  You can check that the installation was sucessful by following the instruction [here](https://drake.mit.edu/python_bindings.html#using-the-python-bindings).

We have used Mosek to solve most of the examples. To solve using Mosek, you'll need to give Drake access to a Mosek license file as described [here](https://drake.mit.edu/bazel.html#mosek). Mosek provides a personal academic license for free.

You will also need to install `gcs` and its dependencies. You can do this by running
```
pip install -r requirements.txt
```

### Running Examples
Once all the dependencies have been installed, you can run the examples with jupyter notebooks which can be launched by calling
```
jupyter-notebook
```
from inside this repository.

### Running the Sampling Based Comparison
If you want to compare GCS to sampling based planners (such as PRM), you'll need to install a custom fork of drake that includes bindings for sampling based planners.  To do this run the following, including any of the proprietary solvers you have access to.

```
git clone -b gcs2 git@github.com:mpetersen94/drake.git
mkdir drake-build
cd drake-build
cmake -DWITH_MOSEK=ON [-DWITH_GUROBI=ON -DWITH_ROBOTLOCOMOTION_SNOPT=ON] ../drake
make -j
```

Then add the built bindings to your Python Path using

For Ubuntu 20.04:
```
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.8/site-packages:$PYTHONPATH
```

For macOS:
```
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.9/site-packages:$PYTHONPATH
```
