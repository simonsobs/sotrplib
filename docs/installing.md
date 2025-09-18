# Installing SOTRPLib

SOTRPLib is a python package that provides executables.

## Development requirements

To get ready for development, create a virtual enviroment and install the package. Either use uv (which you can pip install):
```
uv venv --python=3.12
source .venv/bin/activate
git clone https://github.com/simonsobs/sotrplib.git
cd sotrplib
uv pip install -e ".[dev]"
pre-commit install
```
or venv more directly:
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
git clone https://github.com/simonsobs/sotrplib.git
cd sotrplib
pip install -e ".[dev]"
pre-commit install
```
We use `ruff` for formatting. When you go to commit your code, it will automatically be 
formatted thanks to the pre-commit hook. It's important to follow all of the above steps
to ensure you have all of the available requirements.

Tests are performed using `pytest`.