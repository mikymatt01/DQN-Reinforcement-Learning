FROM python:3.9

WORKDIR /code/
#to COPY the remote file at working directory in container
COPY test2.* /code/
# Now the structure looks like this '/usr/app/src/test.py'
COPY requirements.txt /code/

#CMD instruction should be used to run the software
RUN pip install -r requirements.txt
#contained by your image, along with any arguments.

CMD [ "python3.9", "./test2.py"]
