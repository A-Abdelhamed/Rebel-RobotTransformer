FROM tensorflow/tensorflow:latest-gpu

# Set working directory inside the container
WORKDIR /app

# Copy all repo files into /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
