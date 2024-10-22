FROM gcc:latest   
RUN apt-get update && apt-get install -y \
        make \
        cmake \  
        git \  
        libopenblas-dev \ 
        liblapack-dev \
        liblapacke-dev \
        && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/share/lapack  
COPY lapackCholesky.cpp /usr/share/lapack/lapackCholesky.cpp
RUN g++ -o lapack lapackCholesky.cpp -lopenblas -llapacke -lm -lpthread
ENTRYPOINT ["./lapack"] 
