FROM apache/airflow:3.1.3

# Create models directory
USER root
RUN mkdir -p /app/models

# Copy requirements.txt to the image
COPY requirements.txt /tmp/requirements.txt

# Switch to airflow user BEFORE pip install
USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Switch back to root only if needed (optional)
USER root
RUN rm /tmp/requirements.txt

# Return to airflow user at the end
USER airflow

###usarse por si acaso, como airflow solo orquesta no necesita la imagen customizada####