## Process this file with automake to produce Makefile.in

SUBDIRS = src figures . 

info_TEXINFOS = unuran.texi unuran_win32.texi

# Location of same source files 
TEXINFO_TEX = ./src/texinfo.tex

AM_MAKEINFOFLAGS = \
 -I $(top_srcdir)/doc \
 -I $(top_srcdir)/doc/src \
 --no-split

AM_MAKEINFOHTMLFLAGS = \
 -I $(top_srcdir)/doc \
 -I $(top_srcdir)/doc/src \
 --css-include=$(top_srcdir)/doc/src/unuran.css \
 --no-split

# Automatically created part of manual
unuran_src = ./src/unuran_src.texi

# Files with examples.
# These example files are generated from the corresponding
# files $(top_srcdir)/examples/example*.c
unuran_examples = \
 ./src/ref_example0.texi	./src/ref_example0_str.texi \
 ./src/ref_example1.texi	./src/ref_example1_str.texi \
 ./src/ref_example2.texi	./src/ref_example2_str.texi \
 ./src/ref_example3.texi	./src/ref_example3_str.texi \
 ./src/ref_example_reinit.texi \
 ./src/ref_example_gsl.texi \
 ./src/ref_example_rngstreams.texi \
 ./src/ref_example_anti.texi	./src/ref_example_anti_str.texi \
 ./src/ref_example_cext.texi \
 ./src/ref_example_cont.texi	./src/ref_example_cont_str.texi \
 ./src/ref_example_dext.texi \
 ./src/ref_example_discr.texi	./src/ref_example_discr_str.texi \
 ./src/ref_example_emp.texi	./src/ref_example_emp_str.texi \
 ./src/ref_example_vemp.texi	./src/ref_example_FuncStr.texi \
 ./src/ref_example_mixt.texi    ./src/ref_example_mixt_inv.texi

# Sources for UNU.RAN manual
unuran_TEXINFOS = \
 unuran.texi \
 ./src/methods_cont.texi \
 ./src/methods_discr.texi \
 ./src/methods_cemp.texi \
 ./src/methods_cvec.texi \
 ./src/methods_mcmc.texi \
 ./src/methods_cvemp.texi \
 ./src/methods_matr.texi \
 $(unuran_src) \
 $(unuran_examples) \
 $(top_srcdir)/doc/figures/*.eps \
 $(top_srcdir)/doc/figures/*.pdf \
 $(top_srcdir)/doc/figures/*.png \
 $(top_srcdir)/doc/figures/*.txt


# Sources for (short) Win32 manual

# Files with examples.
# These example files are generated from the corresponding
# files $(top_srcdir)/scripts/win32/example?.c
unuran_win32_examples = \
 ./src/ref_example_win32_1.texi \
 ./src/ref_example_win32_2.texi

unuran_win32_TEXINFOS = \
 unuran_win32.texi \
 unuran_win32_libname.texi \
 $(unuran_win32_examples)


noinst_DATA = 

EXTRA_DIST = \
 unuran.html \
 unuran.txt \
 unuran.pdf \
 unuran_win32.html \
 unuran_win32.pdf


# Generated texi files ...
sinclude ./src/.dep-unuran_src_texi
$(unuran_src): $(top_srcdir)/doc/src/make_texi.pl
	$(top_srcdir)/doc/src/make_texi.pl $(top_srcdir) > $(unuran_src)

# Generated texi files for examples ...
$(unuran_examples): $(top_srcdir)/examples/example*.c
	sed 's/{/@{/g;s/}/@}/g' < $(top_srcdir)/examples/`echo $* | sed 's/src\/ref_//'`.c > $@

$(unuran_win32_examples): $(top_srcdir)/scripts/win32/example*.c
	sed 's/{/@{/g;s/}/@}/g' < \
		$(top_srcdir)/scripts/win32/`echo $* | sed 's/src\/ref_//' | sed 's/_win32_//'`.c > $@

# Rules for creating info, html, dvi, ps, and pdf files are provided by automake.
# Add new rules for txt.
unuran.txt: unuran.texi version.texi $(unuran_TEXINFOS)
	$(MAKEINFO) $(AM_MAKEINFOFLAGS) --fill-column=80 --no-headers --number-sections \
		unuran.texi -o unuran.txt

txt: unuran.txt

# Name of Win32 libfile
unuran_win32_libname.texi:
	@echo -n "@set LIBUNURAN libunuran" > unuran_win32_libname.texi
	@grep PACKAGE_VERSION $(top_srcdir)/config.h | \
		sed 's/.*_VERSION\s*\"\(.*\)\..*\"/\1/' | sed 's/\.//' \
	>> unuran_win32_libname.texi
	@echo -n "@set ZIPFILENAME unuran-" >> unuran_win32_libname.texi
	@grep PACKAGE_VERSION $(top_srcdir)/config.h | \
		sed 's/.*_VERSION\s*\"\(.*\)\"/\1-win32.zip/' \
	>> unuran_win32_libname.texi

# Clean backup and temp files
CLEANFILES = \
 *~

# Clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in \
 mdate-sh \
 $(unuran_src) \
 $(unuran_examples) \
 $(unuran_win32_examples) \
 unuran_win32_libname.texi \
 unuran.txt \
 \
 $(top_srcdir)/aclocal.m4 \
 $(top_srcdir)/config.h.in \
 $(top_srcdir)/configure

# Remark: the last three lines are an ugly hack
# to remove these files in the top source directory.
# It cannot be done in $(top_srcdir)/Makefile.am as
# this breaks the cleaning process.
