#!/usr/bin/perl
##############################################################################
#                                                                            #
#           UNURAN -- Universal Non-Uniform Random number generator          #
#                                                                            #
##############################################################################
#                                                                            #
#   FILE:    remove_comments.pl                                              #
#                                                                            #
#   Remove all comments and blank lines from given C source or header file   #
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
usage: $progname <file> 
      
Removes all comments and blank lines from C file <file>

EOM

    exit -1;
}

############################################################

# year
my $year = 1900 + (localtime(time))[5];

# read file name from argument list ...
my $file = shift;
(usage and die) unless $file;
print "stripping $file ...\n";

# open file ...
open IN, $file or die "cannot open file $file\n";
    
# read file ...
my $content = '';
while (<IN>) { $content .= $_; }
close IN;

# some comments must not be removed.
# so we first masks them
$content =~ s {/\*\s*FALLTHROUGH\s*\*/} [\@\@FALLTHROUGH\@\@]gsx;

# remove all comments and empty lines ...
$content =~ s {/\*.*?\*/} []gsx;
$content =~ s /\n\s*\n/\n/gsx;

# now unmask these special comments
$content =~ s {\@\@FALLTHROUGH\@\@} [/\* FALLTHROUGH \*/]gsx;

# copyright statment
my $copyright =
    "/* Copyright (c) 2000-$year Wolfgang Hoermann and Josef Leydold */\n" .
    "/* Department of Statistics and Mathematics, WU Wien, Austria  */\n";

# print into file
open OUT, ">$file";
print OUT $copyright;
print OUT $content;
close OUT;

# end
exit 0;

############################################################
