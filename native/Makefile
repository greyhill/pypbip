PYTHON=/usr/include/python2.7
PYSHARED=/usr/include/pyshared
LAPACK=/usr/include/atlas

CC=gcc
CFLAGS=-g3 -Wall -Wextra -I${PYTHON} -fPIC -I${LAPACK} -fopenmp

OFILES=pypbip_omp.o \
			 pypbip_native.o
OUT=pypbip_native.so

${OUT}: ${OFILES}
	${CC} ${CFLAGS} -o $@ $^ -lcblas -shared

clean:
	${RM} ${OUT} ${OFILES}

