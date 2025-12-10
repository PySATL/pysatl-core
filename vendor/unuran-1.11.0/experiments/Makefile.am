## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
 -I$(top_srcdir)/src \
 -I$(top_srcdir)/src/tests

AM_LDFLAGS = \
 -L$(top_builddir)/src

LDADD = \
 $(top_builddir)/src/libunuran.la

noinst_PROGRAMS = \
 make_timing_table

EXTRA_DIST = \
 timings_chap_8.conf \
 timings_chap_10.conf

# clean backup files
CLEANFILES = \
 unuran.log \
 *~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in
