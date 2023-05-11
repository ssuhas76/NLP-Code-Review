FROM python:3.11.2

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install nltk
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

COPY ./app /code/app

EXPOSE 8000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]