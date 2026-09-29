# dolfinx/dolfinx:nightly on Aug 26th 2026
FROM dolfinx/dolfinx@sha256:3fd59c6a504b3d7e4dc5d3c191de6d16953ec44f9f60ca8c6480e284af8b36b9

ENV PYTHONPYCACHEPREFIX=/root/.cache/cpython

ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_JUPYTER_BACKEND="html"
ENV LIBGL_ALWAYS_SOFTWARE=0

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Make CppJIT compile with optimizations. C++23 is required by running_error.h
# (std::float16_t, <stdfloat>); CppInterOp otherwise defaults to -std=c++17.
ARG EXTRA_INTERPRETER_ARGS="-O3 -std=c++23"
ENV CPPINTEROP_EXTRA_INTERPRETER_ARGS=${EXTRA_INTERPRETER_ARGS}

# Install system dependencies
COPY apt-requirements.txt /tmp/apt-requirements.txt
RUN apt-get -y update && \
    xargs -a /tmp/apt-requirements.txt apt-get -y install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install prebuild wheels maintained at https://gitlab.com/uniluxembourg/fstm/doe/zed/tools/python/wheels
RUN pip install --no-cache-dir --index-url https://gitlab.com/api/v4/projects/59503118/packages/pypi/simple \
    vtk==9.6.2 && \
    find / -type f -name 'allDict.cxx*' -delete

# Install python dependencies for efficient rebuilds.
# Note: dolfiny itself is installed with --no-deps below, so its dependencies have to be listed here.
RUN pip install --no-cache-dir -v \
    coverage \
    git+https://github.com/fenics-dolfiny/ffcx-backends \
    git+https://github.com/fenics-dolfiny/ufl-units \
    imageio \
    jupyter \
    "jupyter-book>=2.0.0" \
    jupytext \
    matplotlib \
    mypy \
    numba \
    pdbpp \
    pygraphviz \
    pytest-xdist \
    "pyvista[jupyter]>=0.45.0" \
    ruff \
    scipy \
    sympy \
    typos \
    cppjit

# Setup Dolfiny
COPY . /src/dolfiny/

RUN cd /src/dolfiny && \
    pip install "." -v --no-build-isolation --no-deps
