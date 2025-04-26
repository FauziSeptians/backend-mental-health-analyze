# Gunakan base image Python
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Copy semua file project ke dalam container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port Flask app (misalnya port 4000)
EXPOSE 4000

# Jalankan aplikasi
CMD ["python", "app/main.py"]
