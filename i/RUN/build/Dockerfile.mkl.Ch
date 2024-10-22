FROM intel/oneapi-basekit:latest 
COPY mklCholesky.cpp /usr/share/mkl/mkl.cpp
WORKDIR /usr/share/mkl  
RUN apt-get update -y && apt-get install -y clang 
# RUN icpx -mkl -o mkl mkl.cpp -lpthread
RUN icpx -o mkl mkl.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl -fopenmp
# RUN clang++ -o mkl mkl.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
ENTRYPOINT ["./mkl"]
