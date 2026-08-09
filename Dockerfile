FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (Docker layer caching: deps change rarely,
# so this layer is reused across code changes = faster rebuilds).
COPY pyproject.toml .
RUN pip install --no-cache-dir pydantic requests python-dotenv pytest ollama

# Copy the harness package and tests.
COPY harness/ ./harness/
COPY tests/ ./tests/
COPY examples/ ./examples/

# Install the harness package itself.
RUN pip install --no-cache-dir -e .

# Default command: run the test suite (proves the harness works in-container).
CMD ["pytest", "-v"]