# We will use Docker base image from tensorflow
FROM python:3.10.6-buster

WORKDIR /prod

# Requirements_prod will be copied to our Docker container as its requirements.txt
# so it should only have the packages needed for our application.

COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY flood_prediction flood_prediction
COPY setup.py setup.py
RUN pip install .

CMD uvicorn flood_prediction.api.fast:app --host 0.0.0.0 --port $PORT
