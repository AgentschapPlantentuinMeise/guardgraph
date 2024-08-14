# syntax=docker/dockerfile:1
FROM python:3.12-rc-bullseye
WORKDIR /mbg
RUN mkdir -p /mbg/instance/{ix,cubes}
ENV FLASK_APP=guardgraph
ENV FLASK_RUN_HOST=0.0.0.0
#RUN apk add --no-cache gcc musl-dev linux-headers # g++ py3-numpy
# for graphdatascience dependency
# https://arrow.apache.org/install/
RUN apt-get update && apt-get install -y cmake
RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short \
| tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release \
--codename --short).deb
RUN apt install -y -V \
./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V libarrow-dev
RUN apt install -y -V libarrow-flight-dev
# https://arrow.apache.org/docs/developers/guide/step_by_step/building.html
# https://arrow.apache.org/docs/developers/python.html#build-pyarrow
RUN PYARROW_WITH_FLIGHT=1 pip install --no-cache-dir pyarrow
# scikit-learn
RUN apt-get install -y build-essential gfortran libopenblas-dev
# Openstreetmaps
RUN apt-get install -y libxml2-dev libxslt-dev python3-dev
RUN pip install cython==0.29.34 #TODO check when can be done with newer version
RUN pip install git+https://github.com/lxml/lxml
# omspythontools in requirements

# Guarden code
RUN mkdir /repos
RUN cd /repos && git clone https://github.com/neo4j/graph-data-science-client.git
#graphdatascience
RUN cd /repos/graph-data-science-client && pip install .
# Copy guardgraph package
COPY . .
RUN pip install .
RUN pip install phonenumbers email_validator #TODO remove after Flask_IAM update
EXPOSE 5000
CMD ["flask", "run"]
