FROM python:3.11

WORKDIR /api/
COPY . .
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade  -r requirements.txt
COPY boto3-1.28.21-py3-none-any.whl /api/
COPY botocore-1.31.21-py3-none-any.whl /api/

RUN pip install  /api/boto3-1.28.21-py3-none-any.whl
RUN pip install /api/botocore-1.31.21-py3-none-any.whl
RUN pip install uvicorn
# RUN /api/app.py

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "15400" ]