FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y git python3.11 python3.11-venv build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /work
COPY pyproject.toml poetry.lock ./
RUN poetry install --with gpu --no-interaction --no-ansi
COPY . .

ENTRYPOINT ["poetry","run","python","-m","forecast_cli.training.train","--config","conf/linux_4090_train.yaml"] 