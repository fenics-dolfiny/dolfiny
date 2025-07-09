FROM dolfinx/dolfinx:nightly

ARG EXTRA_CLING_ARGS="-O2"
ARG PIP_INDEX_URL="https://gitlab.com/api/v4/projects/59503118/packages/pypi/simple"
ARG NODE_VERSION=22

ENV EXTRA_CLING_ARGS=${EXTRA_CLING_ARGS}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PYTHONPYCACHEPREFIX=/root/.cache/cpython

ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"
ENV PYVISTA_JUPYTER_BACKEND="html"
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV DISPLAY=:99

ENV NVM_DIR=/root/.nvm

# Install system dependencies
RUN apt-get -y update && \
    apt-get -y install \
    libosmesa6 libgl1-mesa-dev xvfb curl \
    libgraphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Node Version Manager (nvm) and Node.js
# See https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-in-docker-for-cicd-jobs
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
RUN bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION"
ENTRYPOINT ["bash", "-c", "source $NVM_DIR/nvm.sh && exec \"$@\"", "--"]
CMD ["/bin/bash"]

# Install prebuild wheels maintained at https://gitlab.com/uniluxembourg/fstm/doe/zed/tools/python/wheels
RUN pip install \
        cppyy-cling==6.32.8 \
        vtk==9.4.2

# Setup Dolfiny
ADD . /src/dolfiny/

RUN cd /src/dolfiny \
    && \
    pip install ".[all]" -v \
    && \
    find / -type f -name 'allDict.cxx*' -delete \
    && \
    python3 -c 'import cppyy'
