FROM python:3.12.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir poetry==2.2.1
WORKDIR /app
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry sync --only main --no-root
COPY src ./src
RUN poetry install --only main
ENTRYPOINT ["poetry", "run", "habit-ml"]
CMD ["demo"]
