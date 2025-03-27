FROM --platform=linux/amd64 python:3.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-setuptools \
    git \
    gcc \
    g++ \
    build-essential \
    curl \
    gnupg \
    wget \
    unzip \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install Bazel
RUN wget -O bazel-5.4.0-installer-linux-x86_64.sh https://github.com/bazelbuild/bazel/releases/download/5.4.0/bazel-5.4.0-installer-linux-x86_64.sh && \
    chmod +x bazel-5.4.0-installer-linux-x86_64.sh && \
    ./bazel-5.4.0-installer-linux-x86_64.sh && \
    rm bazel-5.4.0-installer-linux-x86_64.sh

# Set up your working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install trimmed_match
RUN git clone https://github.com/google/trimmed_match.git /opt/trimmed_match && \
    cd /opt/trimmed_match && \
    PYTHON_BIN_PATH=$(which python) python -m pip install .

# Copy your project files
COPY . /app/

# Install your own package
RUN pip install -e .

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port for Jupyter
EXPOSE 8888

# Entry point
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["jupyter notebook --ip=0.0.0.0 --no-browser --allow-root"]