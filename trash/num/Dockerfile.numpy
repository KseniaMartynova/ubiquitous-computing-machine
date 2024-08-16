FROM python:3.9
RUN apt-get update && apt-get upgrade && apt-get install nano && apt-get install sudo
RUN pip install numpy
COPY NumPy.py /usr/share/num/NumPy.py
WORKDIR /usr/share/num
RUN python3 NumPy.py 
CMD ["python", "main.py"]
