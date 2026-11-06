FROM pytorch/pytorch:2.4.0-cuda12.1-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GLASSBOX_DIR=/glassbox \
    BLACKBOX_DIR=/glassbox \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY pyproject.toml ./
COPY glassbox/ ./glassbox/
COPY blackbox/ ./blackbox/

RUN pip install --no-cache-dir -e ".[dev,search]" || pip install --no-cache-dir -e .

VOLUME ["/glassbox", "/weights"]

ENTRYPOINT ["glassbox"]
CMD ["--help"]
