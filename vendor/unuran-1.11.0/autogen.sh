#! /bin/sh
# Run this to generate all the initial makefiles, etc.

# Constants
srcdir=`dirname $0`
test -z "$srcdir" && srcdir=.

ORIGDIR=`pwd`
cd $srcdir
PROJECT=unuran
TEST_TYPE=-f
FILE=src/unuran_config.h          # a file that should exist in the source dir
export WANT_AUTOMAKE=1.16

# Check for required programs
(libtoolize --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have libtool installed to compile $PROJECT."
	exit 1
}

(autoheader --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have autoheader installed to compile $PROJECT."
	exit 1
}

(autoconf --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have autoconf installed to compile $PROJECT."
	exit 1
}

(automake --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have automake installed to compile $PROJECT."
	exit 1
}

test $TEST_TYPE $FILE || {
	echo "You must run this script in the top-level $PROJECT directory"
	exit 1
}

# Check for given command line arguments
if test -z "$*"; then
	echo "I am going to run ./configure with no arguments - if you wish "
	echo "to pass any to it, please specify them on the $0 command line."
	echo
fi

# Set flags for special compilers
case $CC in
*xlc | *xlc\ * | *lcc | *lcc\ *) am_opt=--include-deps;;
esac

# Run autotools
echo "Running libtoolize ..."
libtoolize --automake

echo; echo "Running aclocal ..."
aclocal

echo; echo "Running autoheader ..."
autoheader --warnings=all

echo; echo "Running automake ..."
automake --warnings=all --add-missing $am_opt

echo; echo "Running autoconf ..."
autoconf --warnings=all
echo

# Change to original directory
cd $ORIGDIR

# Run ./configure script
echo; echo "Running configure ..."
$srcdir/configure --enable-maintainer-mode --enable-deprecated --enable-check-struct --enable-logging "$@"

# End
echo 
echo "Now type 'make' to compile $PROJECT."
echo 

exit 0
