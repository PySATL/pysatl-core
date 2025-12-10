## Process this file with automake to produce Makefile.in

noinst_PROGRAMS = compute_machine_constants

EXTRA_DIST = \
 make_urng_header.pl \
 merge_h.pl \
 read_PDF.pl \
 remove_comments.pl \
 run_valgrind.sh \
\
 win32/Makefile.win32 \
 win32/build.sh \
 win32/make_dll_def.pl \
 win32/example1.c \
 win32/example2.c

noinst_HEADERS = \
  $(top_srcdir)/src/utils/unur_fp_const_source.h

# create files with floating point constants
$(top_srcdir)/src/utils/unur_fp_const_source.h: compute_machine_constants
	test -f $(top_srcdir)/src/utils/unur_fp_const_source.h \
	|| ./compute_machine_constants > $(top_srcdir)/src/utils/unur_fp_const_source.h

# clean backup files
CLEANFILES = \
 *~ \
 win32/*~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in
