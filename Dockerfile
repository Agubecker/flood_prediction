FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY flood_prediction flood_prediction
COPY model model
COPY setup.py setup.py

RUN pip install .

CMD uvicorn flood_prediction.api.fast:app --host 0.0.0.0 --port $PORT
