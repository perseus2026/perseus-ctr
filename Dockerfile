FROM python:3.12-slim
WORKDIR /opt/perseus
COPY pyproject.toml README.md LICENSE ./
COPY perseus/ ./perseus/
RUN python -m pip install --no-cache-dir .
COPY examples/ ./examples/
CMD ["python", "-m", "perseus", "--help"]
