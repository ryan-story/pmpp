nvcc -Xcompiler -fPIC -shared -o conv2dcuda_wrapper.so conv2d.cu 
nvcc -Xcompiler -fPIC -shared -o libcublas_wrapper.so cublas_wrapper.c -lcublas