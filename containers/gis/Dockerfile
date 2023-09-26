# https://towardsdatascience.com/configuring-a-minimal-docker-image-for-spatial-analysis-with-python-dc9970ca8a8a
# Official GDAL image as base image
FROM osgeo/gdal:ubuntu-small-latest

# install pip
RUN apt-get update && apt-get -y install python3-pip --fix-missing

# Set the working directory in the container
WORKDIR /mbg

# Copy the requirements.txt file to the container
COPY requirements.txt /mbg/

# Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt
