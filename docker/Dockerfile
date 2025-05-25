FROM dolfinx/dolfinx:nightly

ARG EXTRA_CLING_ARGS="-O2"
ARG PIP_INDEX_URL="https://gitlab.com/api/v4/projects/59503118/packages/pypi/simple"

ENV EXTRA_CLING_ARGS=${EXTRA_CLING_ARGS}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PYTHONPYCACHEPREFIX=/root/.cache/cpython

# Install system dependencies
RUN apt-get -qq update && \
    apt-get -y install libosmesa6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install prebuild wheels maintained at https://gitlab.com/uniluxembourg/fstm/doe/zed/tools/python/wheels
RUN pip install \
        cppyy-cling==6.32.8 \
        vtk==9.4.2

ADD . /src/dolfiny/

RUN cd /src/dolfiny \
    && \
    pip3 install .[all] -v \
    && \
    find / -type f -name 'allDict.cxx*' -delete \
    && \
    python3 -c 'import cppyy'
