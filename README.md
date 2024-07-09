# Motion Planning around Obstacles with Convex Optimization

This repository contains the code to reproduce the examples in the paper [Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/abs/2205.04422) by Tobia Marcucci, Mark Petersen, David von Wrangel, and Russ Tedrake.

## Recommended: Mature implementation in Drake.

Since the publication of this paper, we have made a mature software
implementation (with numerous improvements) of the [GCS Trajectory Optimization
algorithm](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_gcs_trajectory_optimization.html)
available in [Drake](https://drake.mit.edu). Most all of the examples from this
paper have been reproduced and improved using the Drake implementation; they
can be found linked from [these course
notes](https://underactuated.mit.edu/optimization.html#gcs).

To use GCS Trajectory Optimization in your own code,  we recommend that you use the version in Drake which is being actively developed and improved.

## Running via Deepnote
Most of the examples and reproductions can be run on [Deepnote](https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/GCS-Motion-Planning-around-Obstacles-with-Convex-Optimization-3e7290e8-b92c-4efc-9b58-28a724a78142).

After duplicating the project into your own account, be sure to run the `MosekLicenseUpload.ipynb` notebook to make your Mosek License available for solving the optimization problems.

Note: The Bimanual reproductions do not yet work on Deepnote and the UAV and Maze reproductions have been shrunk in size to avoid hitting memory limits on Deepnote.

## Running locally

### Option 1: Docker
We provide a [dockerfile](docker/deepnote/Dockerfile) with a custom build of [Drake](https://drake.mit.edu) that entails sampling based planners for the comparison. Note that the docker does not include a build of Gurobi, which has been only used in the Iris region generation.

Pull the docker:
```sh
docker pull wrangelvid/drake:gcs-science-robotics
```

Run the docker:
```sh
docker run -i -p 7000:7000 -p 8888:8888 -w /gcs-science-robotics -t wrangelvid/drake:gcs-science-robotics
```

In another shell, copy over your mosek license:

```sh
docker cp [PATH_TO_MOSEK.lic] [container_id]:/tmp/mosek.lic
```

Once the docker has been build and run, you can run the examples with jupyter notebooks:

In the docker run the jupyter server:
```sh
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
```

On your machine go to http://localhost:8888/ You will find the reproduction notebooks in the reproduction folder.

Note: The instructions here use port 7000 for meshcat (robot visualization) and 8888 for jupyter. If these ports are already in use on your machine, you can change the `-p` flag in the `docker run` command above; for instance use `-p 7001:7000` to map the docker port 7000 to your localhost 7001 for meshcat.

### Option 2: Local Installation
If you want to compare GCS to sampling based planners (such as PRM), you'll need to install a custom fork of [Drake](https://drake.mit.edu) that includes bindings for sampling based planners.  To do this run the following, including any of the proprietary solvers you have access to. You may build it with Gurobi.

**Note**: This branch is not being actively maintained and will become more and more difficult to build as the surrounding libraries continue to evolve. To use Graphs of Convex Sets (GCS) Trajectory Optimization in your own code, we recommend that you use the version in Drake which is being actively developed and improved.

```
git clone -b gcs-science-robotics git@github.com:wrangelvid/drake.git
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

We have used Mosek to solve most of the examples. To solve using Mosek, you'll need to give Drake access to a Mosek license file as described [here](https://drake.mit.edu/bazel.html#mosek). Mosek provides a personal academic license for free.

You will also need to install `gcs-science-robotics` and its dependencies. From inside this repository, run the following:
```
pip install -e .
```

Once all the dependencies have been installed, you can run the examples with jupyter notebooks which can be launched by calling
```
jupyter-notebook
```
from inside this repository.
