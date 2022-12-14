# The use TOPSRCDIR allows to compile this testsuite for a clone (and different revision)
# of this repository. If the environment variable topsrcdir is the top source directory
# of the clone, compile this testsuite as:
#
# defines="-Dmzd_randomize=bench_randomize"
# if ! grep lesser_LSB $topsrcdir/src/misc.h >/dev/null; then
#   defines="$defines -DBENCH_RANDOM_REVERSE"
# fi
# if ! grep rci_t $topsrcdir/src/misc.h >/dev/null; then
#   defines="$defines -Drci_t=size_t -Dwi_t=size_t"
# fi
# make bench_elimination bench_multiplication TOPSRCDIR="$topsrcdir" DEFINES="$defines"
#
# Note that if (once) the revision of TOPSRCDIR is new enough and contains the
# current fast mzd_randomize (or when you don't care that it's slow) then you
# can completely leave out DEFINES and just compile as:
#
# make TOPSRCDIR="$topsrcdir"
#
# Finally, if you also leave out TOPSRCDIR and just run 'make' then the testsuite
# is compiled against the current source tree.

CC = gcc -std=gnu99
DEFINES =
TOPSRCDIR = ..
CFLAGS = -I$(TOPSRCDIR) -D_XOPEN_SOURCE=600 -g -O2 $(DEFINES) 
STAGEDIR := $(realpath -s $(TOPSRCDIR)/.libs)
LDFLAGS = -L$(STAGEDIR) -Wl,-rpath,$(STAGEDIR) -lm4ri -lm
PAPI_FLAGS =  
PAPI_CFLAGS = 
DEBUG = -ggdb

TEST_PRGS = \
	test_elimination \
	test_multiplication \
	test_trsm \
	test_ple \
	test_solve \
	test_kernel \
	test_random \
	test_smallops \
	test_transpose \
	test_colswap \
	test_misc \
	test_invert

BENCH_PRGS = \
	bench_elimination \
	bench_multiplication \
	bench_ple \
	bench_trsm \
	bench_elimination_sparse \
	bench_mzd \
	bench_invert

M4RI_HEADERS = \
        ../src/config.h \
	../src/m4ri.h \
	../src/brilliantrussian.h \
        ../src/echelonform.h \
        ../src/graycode.h \
        ../src/misc.h \
        ../src/mmc.h \
        ../src/parity.h \
        ../src/ple.h \
        ../src/ple_russian.h \
        ../src/solve.h \
        ../src/strassen.h \
        ../src/triangular.h \
        ../src/triangular_russian.h \
        ../src/mzd.h \
        ../src/mzp.h \
        ../src/xor.h

CPUCYCLES_DIR = ./cpucycles-20060326

.PHONY: all clean dist-clean

all: $(TEST_PRGS) $(BENCH_PRGS)

bench_%: cpucycles.o benchmarking.o bench_%.c Makefile cpucycles.h $(M4RI_HEADERS)
	$(CC) $(DEBUG) $(PAPI_CFLAGS) $(CFLAGS) $@.c cpucycles.o benchmarking.o $(LDFLAGS) $(PAPI_FLAGS) -lm -o $@

test_%: test_%.c testing.o Makefile $(M4RI_HEADERS)
	$(CC) $(DEBUG) $(CFLAGS) $@.c $(LDFLAGS) testing.o -o $@

clean:
	rm -f $(TEST_PRGS)
	rm -f $(BENCH_PRGS)
	rm -f *.o cpucycles.h
	rm -f $(CPUCYCLES_DIR)/cpucycles.o
	rm -f $(CPUCYCLES_DIR)/cpucycles.h

distclean: clean
	# rm -f Makefile

cpucycles.h: cpucycles.o

cpucycles.o:
	(cd $(CPUCYCLES_DIR); sh do; cp cpucycles.o ..; cp cpucycles.h ..)

testing.o: testing.c testing.h ../src/misc.h ../src/mzd.h
	$(CC) $(DEBUG) $(CFLAGS) -c testing.c $(LDFLAGS) -o $@

benchmarking.o: benchmarking.c benchmarking.h ../src/config.h ../src/misc.h
	$(CC) $(DEBUG) $(PAPI_CFLAGS) $(CFLAGS) -c benchmarking.c $(LDFLAGS) -o $@

Makefile: # Makefile.in ../config.status
	(cd .. && /bin/sh ./config.status testsuite/Makefile)
