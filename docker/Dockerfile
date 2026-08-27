FROM dolfinx/dolfinx:nightly

ENV PYTHONPYCACHEPREFIX=/root/.cache/cpython

ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_JUPYTER_BACKEND="html"
ENV LIBGL_ALWAYS_SOFTWARE=1

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

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

# Build cppjit from source (no published wheel yet.
RUN pip install --no-cache-dir --no-build-isolation -v \
    "cppjit @ git+https://github.com/compiler-research/cppjit" \
    --config-settings=cmake.define.Clang_DIR=/usr/lib/llvm-21/lib/cmake/clang

# Install python dependencies for efficient rebuilds
RUN pip install --no-cache-dir -v \
    coverage \
    git+https://github.com/fenics-dolfiny/ffcx-backends \
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
    typos

# Setup Dolfiny
COPY . /src/dolfiny/

RUN cd /src/dolfiny && \
    pip install "." -v --no-build-isolation --no-deps
