#!/usr/bin/perl
##############################################################################
#                                                                            #
#           UNURAN -- Universal Non-Uniform Random number generator          #
#                                                                            #
##############################################################################
#                                                                            #
#   FILE:    makedll_def.pl                                                  #
#                                                                            #
#   Read all UNU.RAN header files and create .def file for .dll              #
#                                                                            #
##############################################################################
#                                                                            #
#   Copyright (c) 2000-2007 Wolfgang Hoermann and Josef Leydold              #
#   Department of Statistics and Mathematics, WU Wien, Austria               #
#                                                                            #
#   This program is free software; you can redistribute it and/or modify     #
#   it under the terms of the GNU General Public License as published by     #
#   the Free Software Foundation; either version 2 of the License, or        #
#   (at your option) any later version.                                      #
#                                                                            #
#   This program is distributed in the hope that it will be useful,          #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#   GNU General Public License for more details.                             #
#                                                                            #
#   You should have received a copy of the GNU General Public License        #
#   along with this program; if not, write to the                            #
#   Free Software Foundation, Inc.,                                          #
#   59 Temple Place, Suite 330, Boston, MA 02111-1307, USA                   #
#                                                                            #
##############################################################################

use strict;
use Getopt::Std;

my $DEBUG = 0;

############################################################
# constants

my $DEP_file = ".dep-unuran_def";

############################################################

sub usage {
    my $progname = $0;
    $progname =~ s#^.*/##g;
        
    print STDERR <<EOM;
usage: $progname [-V <version>] [-L <libraryname>] [-I] <file1.h> [ <file2.h> ... ]
      
Scans <file.h> and inserts all header files found in subtree 
rooted at current working directory.
Other header files and those with names containing "config" 
are ignored.
The function declaration and global variables are extracted
and .def file is created.

Option '-V' can be used to set a library version number using
the format 'major.minor'.

Option '-L' can be used to set a library name.

Use option '-I' when also (some) internal functions should be exported.

The output is written on stdout.

EOM

    exit -1;
}

############################################################

# dependencies
my $DEP = "unuran.def: ";

############################################################

use FileHandle;

############################################################
# global variables
	
# default name of library
my $Library = "unuran";

# DATA section (to be set manually!!)
my $DATA = "";
#    "\tunur_errno\tDATA\n";
#    "\tINFINITY\tDATA\n";

# EXPORTS section (functions; set automatically)
my $EXPORTS;
my @EXPORTS;

############################################################

# read options
my %opts;
getopts('V:L:I', \%opts) or usage();
my $Version = $opts{'V'};
$Library = $opts{'L'} if $opts{'L'};
my $INTERNAL = $opts{'I'};

# read master file name from argument list ...
my @master_files;
while ($_ = shift) { push @master_files, $_; }
(usage and die) unless @master_files;

# header files in sub tree (except those containing "config") ...
# (files are stored in an associate array with key=filename and
# value=complete path of file.)
my %header_files;
# included header files ...
my %header_included;

# search subtree for header files ...
open (FILES, "find . |");
while (<FILES>) {
    chomp;
    next unless /^.*\/+(.*\.h)$/;
    next if /config/;
    $header_files{$1} = $_;  # store file and path of file
}
close FILES;

# scan master file(s) ...
foreach my $m (@master_files) { scan_file ($m); }

# write file
print "LIBRARY $Library\n";
print "VERSION $Version\n" if $Version;
print "EXPORTS\n";
print $DATA;
foreach my $e (sort @EXPORTS) { $EXPORTS .= "\t$e\n"; }
print $EXPORTS;

# write dependencies
#open DEP, ">$DEP_file" or die "Cannot open file for writing: $DEP_file";
#print DEP "$DEP\n";
#close DEP;

exit 0;

############################################################

# scan given file ...
sub scan_file {
    my $file = $_[0];
    my $handle = new FileHandle;

    ##    print STDERR "$file\n" if $DEBUG;
    
    # open file ...
    open  $handle, $file or die "cannot find file $file\n";
    
    # read file ...
    my $content = '';
    while (<$handle>) {
	$content .= $_;
    }
    close $handle;

    # remove all comments and empty lines ...
    $content =~ s {/\*.*?\*/} []gsx;
    $content =~ s /\n\s*\n/\n/gsx;

    # split into lines ...
    my @lines = split /\n/, $content;

    my $line;
    foreach my $l (@lines) {
	chomp($l);
	if ($line =~ /^.*\\\s*$/) { 
	    $line =~ s/\\\s*$//;
	    $line .= $l;
	    next;
	}
	$line = $l;
	unless ($line =~ /^\#include\s+[<\"](.*)[>\"]/) {
	    # remove all blank lines 
	    next if $line !~ /\w/;
	    # remove all (other) preprocessor directives
	    next if $line =~ /^\s*\#/;
	    # find all non-functions
	    next if $line !~ /\(.+/;
	    next if $line =~ /^\s*typedef\s+/;
	    # internal functions
	    unless ($INTERNAL) {
	    	next if $line =~ /[\s\*]_unur/;
	    }
	    else {
		# remove internal debugging routines
	    	next if $line =~ /[\s\*]_unur_error_cookies/;
	    	next if $line =~ /[\s\*]_unur_distr_\w+_debug/;
	    } 
	    # line with isolated attribute statement
	    next if $line =~ /^\s*ATTRIBUTE__/;
	    # remove special symbols
	    next if $line =~ /_unur_is(zero|one|fsame)/;
	    # exported functions
	    $line =~ /.*?[\s\*]+(\w+)\s*\(/;
	    push @EXPORTS, $1;
	    next;
	}

	# have found a file to be included ...
	my @tmp = split /\//, $1;
	my $include_file = pop @tmp; 
	print STDERR "$include_file  " if $DEBUG;

	# we do not include header files out of the subtree ...
	unless (defined( $header_files{$include_file} ) ) {
	    print STDERR "file not found in subtree ... skip\n" if $DEBUG;
	    next;
	}

	# we do not include header files for deprecated calls
        if ( $include_file =~ /deprecated/ ) {
	    print STDERR "deprecated ... skip\n" if $DEBUG;
	    next;
	}
	    
	# we include header files only once ...
	if (defined( $header_included{$include_file} ) ) {
	    print STDERR "already included ... skip\n" if $DEBUG;
	    next;
	}

	$header_included{$include_file} .= 1;
	
	# have found header file ...
	print STDERR "to be inserted\n" if $DEBUG;

	# add dependency
        $DEP .= "$header_files{$include_file} ";

	# scan header file ...
	scan_file ($header_files{$include_file});

    }

}

############################################################

############################################################
    
