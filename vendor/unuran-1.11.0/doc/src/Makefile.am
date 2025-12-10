## Process this file with automake to produce Makefile.in

EXTRA_DIST = \
 README \
 texinfo.tex \
 make_texi.pl \
 arvag.dh \
 error_debug.dh \
 examples.dh \
 glossary.dh \
 index.dh \
 intro.dh \
 installation.dh \
 methods.dh \
 misc.dh \
 references.dh \
 stddist.dh \
 top.dh

# do not distribute dependencies for unuran_src.texi
dist-hook:
	rm -f .dep-unuran_src_texi

# clean backup files
CLEANFILES = *~

# clean generated files
MAINTAINERCLEANFILES = \
 Makefile.in \
 .dep-unuran_src_texi
