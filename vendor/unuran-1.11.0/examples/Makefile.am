## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
 $(UNURAN_SUPPORTS_GSL) \
 $(UNURAN_SUPPORTS_PRNG) \
 $(UNURAN_SUPPORTS_RNGSTREAM) \
 -I$(top_srcdir)/src \
 -I$(top_srcdir)/src/uniform

AM_LDFLAGS = \
 -L$(top_builddir)/src

LDADD = \
 $(top_builddir)/src/libunuran.la

EXAMPLES = \
 example0 example0_str \
 example1 example1_str \
 example2 example2_str \
 example3 example3_str \
 example_reinit \
 example_cext \
 example_cont  example_cont_str \
 example_dext \
 example_discr example_discr_str \
 example_emp   example_emp_str \
 example_vemp  \
 example_FuncStr \
 example_mcorr \
 example_mixt  example_mixt_inv \
 example_cpp \
 example_gsl \
 example_rngstreams \
 example_prng  example_prng_str \
 example_anti  example_anti_str \
 example_errorhandler

example_cpp_SOURCES = example_cpp.cpp

check_PROGRAMS = $(EXAMPLES)

TESTS = $(EXAMPLES)

# run tests with valgrind
if HAVE_VALGRIND
valgrind: $(check_PROGRAMS)
	@list='$(check_PROGRAMS)'; for p in $$list; do \
		$(top_srcdir)/scripts/run_valgrind.sh ./$$p; \
	done
endif

# clean log files and backup files
CLEANFILES = \
 unuran.log \
 valgrind-* \
 *~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in
