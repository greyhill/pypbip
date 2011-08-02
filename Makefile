LAPACK=/usr/include/atlas

CC=gcc
CFLAGS=-g3 -Wall -Wextra -fPIC -I${LAPACK} -fopenmp

OFILES=pypbip_omp.o \
			 pypbip_ksvd.o 
OUT=pypbip_native.so

${OUT}: ${OFILES}
	${CC} ${CFLAGS} -o $@ $^ -lcblas -shared

clean:
	${RM} ${OUT} ${OFILES} *.pyc

