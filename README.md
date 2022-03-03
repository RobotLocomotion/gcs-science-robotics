# MP-GCS Motion Planning on Graph of Convex Sets

Code for Motion Planning around Obstacles with Convex Optimization

## Installing Dependencies
This code depends on [Drake](https://drake.mit.edu), specifically its Python bindings. To install the bindings run

```
sudo apt-get update
sudo apt-get install --no-install-recommends \
  ca-certificates gnupg lsb-release wget
wget -qO- https://drake-apt.csail.mit.edu/drake.asc | gpg --dearmor - \
  | sudo tee /etc/apt/trusted.gpg.d/drake.gpg >/dev/null
echo "deb [arch=amd64] https://drake-apt.csail.mit.edu/$(lsb_release -cs) $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/drake.list >/dev/null
sudo apt-get update
sudo apt-get install --no-install-recommends drake-dev
```

To ensure that Python can find the bindings make sure to add drake to the PYTHONPATH.  To do this add one of the following:

For Ubuntu 18.04:
```
export PYTHONPATH=/opt/drake/lib/python3.6/site-packages:$PYTHONPATH
```

For Ubuntu 20.04:
```
export PYTHONPATH=/opt/drake/lib/python3.8/site-packages:$PYTHONPATH
```

For macOS:
```
export PYTHONPATH=/opt/drake/lib/python3.9/site-packages:$PYTHONPATH
```

### Running the Sampling Based Comparison
If you want to compare MP-GCS to sampling based planners (such as PRM), you'll need to install a custom fork of drake that includes bindings for sampling based planners.  To do this run the following, including any of the proprietary solvers you have access to.

```
git clone -b mp-gcs git@github.com:mpetersen94/drake.git
mkdir drake-build
cd drake-build
cmake [-DWITH_GUROBI=ON -DWITH_MOSEK=ON -DWITH_ROBOTLOCOMOTION_SNOPT=ON] ../drake
make -j
```

Then add the built bindings to your Python Path using

For Ubuntu 18.04:
```
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.6/site-packages:$PYTHONPATH
```

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

## Confirming Drake bindings are accessible to Python
To confirm that the Drake bindings are accessible, and that you are using the right set of bindings run

```
python3 -c 'import pydrake.all; print(pydrake.__file__)'
```

and confirm that the printed path matches the expected location of the Python bindings.

## Running Examples
Once all the dependencies have been installed, you can run the examples with jupyter notebooks which can be launched by calling
```
jupyter-notebook
```
from inside this repository.
