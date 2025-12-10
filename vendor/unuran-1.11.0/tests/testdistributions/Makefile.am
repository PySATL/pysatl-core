## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = \
 -I$(top_srcdir)/src

noinst_LTLIBRARIES = libtestdistributions.la

libtestdistributions_la_SOURCES = \
 c_with_multiple_of_pdf.c \
 c_wo_logpdf.c \
 c_w_pdf_from_logpdf.c \
 c_sawtooth_discpdf.c \
 c_sawtooth_contpdf.c \
 vc_w_marginals.c \
 vc_wo_logpdf.c \
 vc_cauchy_RoU_ball.c \
 vc_special_correlations_ar1.c \
 vc_special_correlations_constantrho.c

noinst_HEADERS = testdistributions.h 

# clean backup files
CLEANFILES = *~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in
