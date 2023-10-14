# Define custom function directory
ARG FUNCTION_DIR="/function"

# Use an official Python runtime as a parent image
FROM python:3.11 AS build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}

# Copy function code and models to the docker image from the current folder
COPY app.py ${FUNCTION_DIR}

# Install the function's dependencies
RUN pip install \
    --target ${FUNCTION_DIR} \
        awslambdaric

# Install libxslt and libxml2 development packages, and gcc using apt-get (for Debian-based images)
RUN apt-get update && apt-get install -y libxslt1-dev libxml2-dev gcc

# Use a slim version of the base Python image to reduce the final image size
FROM python:3.11-slim

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

# Copy requirements.txt to the docker image and install dependencies
COPY requirements.txt ${FUNCTION_DIR}
RUN pip install -r ${FUNCTION_DIR}/requirements.txt

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Set the CMD to your handler using the AWS Lambda Runtime Interface Emulator
CMD [ "app.lambda_handler" ]