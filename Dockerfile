FROM python:3.6
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
CMD bash
