FROM dolfinx/dolfinx:nightly

ARG PIP_INDEX_URL="https://gitlab.com/api/v4/projects/59503118/packages/pypi/simple"
ARG EXTRA_CLING_ARGS="-O2"

ADD . /src/dolfiny/

RUN cd /src/dolfiny \
    && \
    pip3 install .[all] \
    && \
    find / -type f -name 'allDict.cxx*' -delete \
    && \
    python3 -c 'import cppyy'

ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV EXTRA_CLING_ARGS=${EXTRA_CLING_ARGS}
ENV PYTHONPYCACHEPREFIX=/root/.cache/cpython
