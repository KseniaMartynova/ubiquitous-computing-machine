FROM python:3.9
RUN apt-get update && apt-get upgrade && apt-get install nano && apt-get install sudo
RUN pip install numpy
COPY NumPy.py .
WORKDIR /usr/share/num
CMD ["python", "main.py"]
