When using con the cloud:

1. Clone this repo
2. You're ready to go!

When using locally:

1. Clone this repo
2. Clone [https://github.com/cloudpendulum/cloudpendulum-local](https://github.com/cloudpendulum/cloudpendulum-local) and follow the installation instructions
3. Sourced in the same venv, call
```
pip install -r requirements.txt
```
4. Register the Jupyter kernel and add to VS Code
```
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name CPenv --display-name "Python (CPvenv)"
```
5. Select the kernel when starting your Jupyter notebook.
6. You're ready to go!
