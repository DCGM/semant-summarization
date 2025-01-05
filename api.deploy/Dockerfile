FROM alpine as git

RUN apk add --no-cache git openssh-client

ARG repo=https://github.com/DCGM/semant-summarization.git
ARG branch=main
ARG now=x

RUN mkdir ~/.ssh && \
    mkdir /tmp/repo && \
    git clone ${repo} /tmp/repo && \
    cd /tmp/repo && \
    git checkout ${branch} && \
    git pull


FROM python:3.10-slim as backend

COPY --from=git /tmp/repo/api/requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
COPY --from=git /tmp/repo/api /app
WORKDIR /app
ENTRYPOINT [ "python", "run.py" ]
