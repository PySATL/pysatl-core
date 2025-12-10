#!/usr/bin/perl
##############################################################################
#                                                                            #
#           UNURAN -- Universal Non-Uniform Random number generator          #
#                                                                            #
##############################################################################
#                                                                            #
#   FILE:    make_urng_header.pl                                             #
#                                                                            #
#   Make header file for UNU.RAN wrapper functions for external URNGs        #
#                                                                            #
##############################################################################
#                                                                            #
#   Copyright (c) 2000-2006 Wolfgang Hoermann and Josef Leydold              #
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

my $DEBUG = 0;

############################################################

sub usage {
    my $progname = $0;
    $progname =~ s#^.*/##g;
        
    print STDERR <<EOM;
usage: $progname <file1> [<file2> ...]
      
Create header file for wrappers for external URNGs 

EOM

    exit -1;
}

############################################################

# year
my $year = 1900 + (localtime(time))[5];

# content of files
my $out = '';

# copyright statment
my $copyright =
    "/*******************************************************************\\\n".
    " *                                                                 *\n".
    " *   UNU.RAN -- Universal Non-Uniform Random number generator      *\n".
    " *                                                                 *\n".
    " *******************************************************************\n".
    " *   Copyright (c) 2000-$year Wolfgang Hoermann and Josef Leydold   *\n".
    " *   Department of Statistics and Mathematics, WU Wien, Austria    *\n".
    "\\*******************************************************************/\n";

my $CPP_header =
    "#undef __BEGIN_DECLS\n".
    "#undef __END_DECLS\n".
    "#ifdef __cplusplus\n".
    "#  define __BEGIN_DECLS extern \"C\" {\n".
    "#  define __END_DECLS }\n".
    "#else\n".
    "#  define __BEGIN_DECLS /* empty */\n".
    "#  define __END_DECLS /* empty */\n".
    "#endif\n".
    "\n".
    "__BEGIN_DECLS\n\n";

my $CPP_bottom =
    "__END_DECLS\n";

# marker
my $marker = '';

# read file name from argument list ...
while (my $file = shift) {

    # open file ...
    open IN, $file or die "cannot open file $file\n";
    
    # do we have to define a marker?
    unless ($marker) {
	$marker = "UNURAN_" . uc($file) . "_SEEN";
	$marker =~ s/\W/\_/g;
    }

    # read file ...
    my $content = '';
    while (<IN>) { $content .= $_; }
    close IN;

    # remove all comments and empty lines ...
    $content =~ s {/\*.*?\*/} []gsx;
    $content =~ s /\n\s*\n/\n/gsx;

    # remove ifdef ...H_SEEN
    if ($content =~ /\#ifndef URNG_.*_H_SEEN/) {
	$content =~ s {\#ifndef\s+URNG_.*_H_SEEN\s+\#define\s+URNG_.*_H_SEEN\s*\n} []gsx;
    	$content =~ s /\#endif\s+$//gsx;
    }

    # add to output
    $out .= $content;
}
    
print $copyright .
    $CPP_header .
    "\#ifndef $marker\n" .
    "\#define $marker\n" .
    $out . 
    "\#endif  /* $marker */\n" .
    $CPP_bottom;

# end
exit 0;

############################################################
