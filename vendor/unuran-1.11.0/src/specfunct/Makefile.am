## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
 -I$(top_srcdir)/src

noinst_LTLIBRARIES = libspecfunct.la

# Routines taken from
# Cephes Math Library Release 2.3:  June, 1995
# by Stephen L. Moshier:
# [ See also README.cephes ]
CEPHES = \
 cephes_source.h \
 cephes_gamma.c \
 cephes_igam.c \
 cephes_incbet.c \
 cephes_ndtr.c \
 cephes_ndtri.c \
 cephes_polevl.c

RMATH =

OTHER = \
 bessel_asympt.c \
 cgamma.c \
 hypot.c \
 log1p.c

libspecfunct_la_SOURCES = $(CEPHES) $(RMATH) $(OTHER)

noinst_HEADERS = \
 unur_specfunct_source.h

# clean backup files
CLEANFILES = *~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in
