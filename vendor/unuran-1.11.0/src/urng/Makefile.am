## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
 -I$(top_srcdir)/src

noinst_LTLIBRARIES = liburng.la

liburng_la_SOURCES = \
 urng_source.h urng_struct.h \
 urng.h urng.c urng_set.c urng_default.c \
 urng_unuran.c

# clean backup files
CLEANFILES = *~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in
