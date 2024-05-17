# Use python as base image
FROM python

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./llama_cpu_server.py /app/llama_cpu_server.py

# Install the needed packages
RUN pip install transformers Flask

# Optional: Define a placeholder environment variable for the Hugging Face access token (if needed)
ENV HUGGINGFACE_HUB_TOKEN="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"

# Expose port 5000 outside of the container
EXPOSE 5000

# Run llama_cpu_server.py when the container launches
CMD ["python", "llama_cpu_server.py"]
