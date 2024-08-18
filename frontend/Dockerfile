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
COPY ./app.py ./app.py
COPY ./auth.py ./auth.py

COPY pages/ pages/

# Expose the application port
EXPOSE 8501

# Add a HEALTHCHECK
## HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]