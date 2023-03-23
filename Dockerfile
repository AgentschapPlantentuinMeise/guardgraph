# syntax=docker/dockerfile:1
FROM python:3.12-alpine
WORKDIR /mbg
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY . .
RUN pip install .
EXPOSE 5000
CMD ["flask", "run"]
