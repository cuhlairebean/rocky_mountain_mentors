# Base image: Miniconda
FROM continuumio/miniconda3

# Copy your environment.yml into the image
COPY environment.yml /tmp/environment.yml

# Create the environment
RUN conda env create -f /tmp/environment.yml

# Make sure conda is initialized
SHELL ["conda", "run", "-n", "rmm-llm", "/bin/bash", "-c"]

# Set working directory inside container
WORKDIR /app

# Copy the rest of your code
COPY . /app

# Ensure the environment is activated by default
ENV CONDA_DEFAULT_ENV=rmm-llm
ENV PATH /opt/conda/envs/rmm-llm/bin:$PATH

# Set entry point
CMD ["conda", "run", "--no-capture-output", "-n", "rmm-llm", "python", "ALICIA.py"]