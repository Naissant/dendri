FROM gitpod/workspace-full

USER gitpod

RUN pip install jupyter jupytext black flake8 pre-commit

RUN pre-commit install
