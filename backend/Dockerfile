FROM python:3.12-slim

# Install Poetry
RUN pip install poetry==1.8.3

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Set the working directory
WORKDIR /code

# Copy Poetry files
COPY ./pyproject.toml ./poetry.lock* ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the application files
COPY ./config ./config
COPY ./src ./src
COPY ./server.py ./server.py

# Expose the application port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
