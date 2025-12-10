## Process this file with automake to produce Makefile.in

# AUTOMAKE_OPTIONS = 

SUBDIRS = . scripts src examples experiments tests doc

DIST_SUBDIRS = $(SUBDIRS)

EXTRA_DIST = \
 README.win32 \
 UPGRADE \
 KNOWN-PROBLEMS \
 autogen.sh


# make special distribution for ROOT
rootdir = tmp-unuran-root
distrootfile = $(PACKAGE)-$(VERSION)-root.tar.gz

remove_rootdir = \
  { test ! -d $(rootdir) \
    || { find $(rootdir) -type d ! -perm -200 -exec chmod u+w {} ';' \
         && rm -fr $(rootdir); }; }

dist-root:
if MAINTAINER_MODE
	@echo "Create tar ball for ROOT" | \
          sed 'h;s/./=/g;p;x;p;x'
	$(remove_rootdir)
	$(MAKE)
	$(MAKE) dist
	mkdir $(rootdir)
	tar zxf $(PACKAGE)-$(VERSION).tar.gz -C $(rootdir) 
	(cd $(rootdir)/$(PACKAGE)-$(VERSION); \
	   sed -e 's/\(AC_INIT(\s*\[unuran\]\s*,\s*\[\)\(.*\)\]\s*,\s*\[/\1\2-root\],\[/' \
		-e 's/\(AM_INIT_AUTOMAKE(\s*\[unuran\]\s*,\s*\[\)\(.*\)\]\s*)/\1\2-root\])/' \
		-e '/^\s*src.*\/Makefile\s*\\/p' -e '/^\s*.*\/Makefile\s*\\/d' \
		< ../../configure.ac > ./configure.ac; \
	   sed -e 's/^\s*SUBDIRS\s*=.*/SUBDIRS = \. src/' \
		< ../../Makefile.am > Makefile.am; \
	   for f in `find ./src -type f -name '*.[ch]' -o -name '*.ch'`; do \
		./scripts/remove_comments.pl $$f; \
	   done; \
	   echo "AC_MSG_NOTICE([ " >> configure.ac; \
	   echo "*===================================================" >> configure.ac; \
	   echo "*" >> configure.ac; \
	   echo "*   This is a pure source distribution!" >> configure.ac; \
	   echo "*" >> configure.ac; \
	   echo "*   For a manual, documented sources, tests," >> configure.ac; \
	   echo "*   and examples please download" >> configure.ac; \
	   echo "*" >> configure.ac; \
	   echo "*       $(PACKAGE)-$(VERSION).tar.gz" >> configure.ac; \
	   echo "*" >> configure.ac; \
	   echo "*===================================================" >> configure.ac; \
	   echo "])" >> configure.ac; \
	   autoreconf; ./configure; \
	   $(MAKE) dist; \
	)
	cp $(rootdir)/$(PACKAGE)-$(VERSION)/$(PACKAGE)-*.tar.gz .
	$(remove_rootdir)
	@echo "$(distrootfile) is ready for distribution" | \
          sed 'h;s/./=/g;p;x;p;x'
else
	@echo "You must enable maintainer mode using"
	@echo "   sh ./configure --enable-maintainer-mode"
	@echo "first to run 'make dist-root'!"
endif

# compile with almost all GCC warnings enabled
if MAINTAINER_MODE
setgccWflags:
	export CFLAGS="`cat $(top_srcdir)/scripts/gccWflags`"; \
	./configure --enable-maintainer-mode

setgccWflags-ansi:
	export CFLAGS="-ansi -pedantic `cat $(top_srcdir)/scripts/gccWflags`"; \
	./configure --enable-maintainer-mode
endif

# run checks in fullcheck mode
fullcheck:
	UNURANFULLCHECK=true $(MAKE) check


# clean backup files
CLEANFILES = *~

# clean generated files
MAINTAINERCLEANFILES = \
 $(srcdir)/Makefile.in \
 $(srcdir)/autoconf/compile \
 $(srcdir)/autoconf/config.guess \
 $(srcdir)/autoconf/config.sub \
 $(srcdir)/autoconf/depcomp \
 $(srcdir)/autoconf/install-sh \
 $(srcdir)/autoconf/ltmain.sh \
 $(srcdir)/autoconf/mdate-sh \
 $(srcdir)/autoconf/missing \
 $(srcdir)/autoconf/test-driver

# Remark: the three files
#   $(top_srcdir)/aclocal.m4
#   $(top_srcdir)/config.h.in
#    $(top_srcdir)/configure
# are removed in doc/Makefile.am.
# It cannot be done here as this breaks the cleaning process.

# auxiliary directories
maintainer-clean-local:
	rm -rf unuran-win32
	@$(remove_rootdir)

