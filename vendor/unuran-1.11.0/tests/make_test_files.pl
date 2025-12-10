#!/usr/bin/perl
##############################################################################
#                                                                            #
#           UNURAN -- Universal Non-Uniform Random number generator          #
#                                                                            #
##############################################################################
#                                                                            #
#   FILE:    make_test_files.pl                                              #
#                                                                            #
#   Read test config file and create C file for tests                        #
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

use English;
use strict;

# ---------------------------------------------------------------------------

sub usage {
    my $progname = $0;
    $progname =~ s#^.*/##g;
    
    print STDERR <<EOM;
usage: $progname <srcdir> <test.conf>
      
  Scans <test.conf>, generates the C source for test routines,
  and writes it on stdout.
  File must have suffix "conf".
      
EOM
      
} # end of usage()

# ---------------------------------------------------------------------------
# read arguments

# src directory
my $src_dir = shift or die "No argument given";
usage and die "$src_dir: No such directory" unless -d $src_dir;

# ---------------------------------------------------------------------------
# Constants

# unuran header file
my $unuran_h_file = "$src_dir/src/unuran.h";

# program name 
my $name_program = $0;
$name_program =~ s#^.*/##g;

# ---------------------------------------------------------------------------
# seed for uniform random number generator.

my $seed = $ENV{SEED} + 0.;
unless ($seed) { $seed = int(rand 1000000) + 1; }

# ---------------------------------------------------------------------------
# Global variables

my %DATA;     # store all data in config file

# special data we want ...
my $method;           # generation method (lower case letters)
my $METHOD;           #                   (upper case letters)
my $gen_type;         # type of generation method
my $distr_type;       # type of distribution
my $C_header_aux;     # part to be added to C header 
my @header_files;     # additional header file included in *.conf

my @othersections = ("new", "set", "get", "chg", "init", "reinit", "sample");

my $test_routines;

my $verify_prototype;
my $verify_routine;

# ---------------------------------------------------------------------------

#############################################################################
#                                                                           #
#  Main                                                                     #
#                                                                           #
#############################################################################

# ---------------------------------------------------------------------------
# read in input file name from argument list ...
# it must hve suffix .conf
my $file_in = shift;
usage and die unless $file_in and $file_in =~ /\.conf$/;

# get base name of file
my $file_name = $file_in;
$file_name =~ s#^.*/##g;
$file_name =~ s/\.conf$//;

# name of log files
my $file_testlog = "$file_name\_test\.log";
my $file_unuranlog = "$file_name\_unuran\.log";

# add unuran header file
push @header_files, $unuran_h_file;

# read data
read_data($file_in);

# scan data and print
scan_main();

# create verify routine (if necessary)
make_unur_set_verify_routine();

# print
print_C_header();
scan_verbatim();
foreach my $s (@othersections) {
    scan_other($s); }
scan_validate();
scan_special();
print_C_routines();
print $verify_routine;
print_C_main();

# ---------------------------------------------------------------------------
# End

exit(0);

#############################################################################
#                                                                           #
#  Read data                                                                #
#                                                                           #
#############################################################################

sub read_data {
    my $file = shift;

    open (CONF, "$file") or die "Cannot open file $file for reading";

    my $section = "start";        # lines before first section are stored in section "start"
    my $subsection = "start";     # lines before first subsection are stored in subsection "start"
    my $header;                   # header of subsection
    my $body;                     # body of subsection
    my $ssc = 0;                  # number of subsection (to store order in input file)

    while (<CONF>) {
	# first remove comments
	next if /^\#/;
	s/(?<!\\)\#.*$//;
	# transform \# --> #
	s/\\\#/\#/g;

	# search for included header files
	if (/^\s*\#\s*include\s+[\"\<](.*)[\"\>]/) {
	    push @header_files, "$src_dir/$1";
	}

	if (/^\s*\[/) {
	    # new (sub)section starts here
	    # save body of last subsection
	    $DATA{$section}[$ssc]{"title"} = $subsection;
	    $DATA{$section}[$ssc]{"header"} = $header;
	    $DATA{$section}[$ssc]{"body"} = $body;
	    
	    ++$ssc;         # increment counter
	    $header = "";   # reset entry of subsection header
	    $body = "";     # reset entry of subsection body
	    
	    if (/^\s*\[\s*(.*?)\s*-\s*(.*?)\s*\:/) {
		# subsection header
		if ($1 ne $section) { die "Invalid subsection $_ in section [$section]"; }
		# update subsection name 
		$subsection = $2;
		# find header of subsection (search for closing "]")
		unless (/\]\s*$/) {
		    until ($header =~ /\]\s*$/) { $header .= "$INPUT_LINE_NUMBER:".<CONF>; }
		    # chop off closing bracket
		    $header =~ s/\]\s*$//;
		}
	    }
	    
	    elsif (/^\s*\[\s*(.*?)\s*\]/) {
		# section header
		if ($DATA{$1}) { die "section already defined"; }
		# update section name and reset subsection name
		$section = $1;
		$subsection = "start";
		$ssc = 0;
	    }
	    
	    else {
		die "Line $INPUT_LINE_NUMBER startet with '[' but is neither section nor subsection";
	    }
	}
	
	else {
	    # subsection body
	    $body .= ($INPUT_LINE_NUMBER-1).":$_";
	}
    }
    close CONF;

    # store last section
    $DATA{$section}[$ssc]{"title"} = $subsection;
    $DATA{$section}[$ssc]{"header"} = $header;
    $DATA{$section}[$ssc]{"body"} = $body;

    # debug_print_data();

} # end of read_data()

# ---------------------------------------------------------------------------

sub debug_print_data {

    foreach my $s (keys %DATA) {
	print "[$s]\n";
	foreach my $subs (@{$DATA{$s}}) {
	    print "[$s \@ ".${$subs}{"title"}."]\n";
	    print ${$subs}{"header"}."\n";
	    print ${$subs}{"body"}."\n";
	}
    }
} # end of debug_print_data()


#############################################################################
#                                                                           #
#  Scan section [main]                                                      #
#                                                                           #
#############################################################################

sub scan_main {

    $DATA{"main"} or die "section [main] missing";

    # scan subsection 
    foreach my $subs (@{$DATA{"main"}}) {
	my $title = ${$subs}{"title"};
	my $body = ${$subs}{"body"};
      MAINSUBS: {
	  if ($title eq "start") {
	      # ignore
	      last MAINSUBS;
	  }
	  if ($title eq "data") {
	      # name of method 
	      if ("\n$body" =~ /\n\d+\:\s*method\s*:\s*(\w+)/) { 
		  $method = $1;
		  $method =~ tr/[A-Z]/[a-z]/;
		  $METHOD = $method;
		  $METHOD =~ tr/[a-z]/[A-Z]/;
	      }
	      last MAINSUBS;
	  }
	  if ($title eq "header") {
	      # additonal part for C header 
	      $C_header_aux = $body;
	      $C_header_aux =~ s/(^|\n)\d+\:/\n/g;   # remove line info
	      last MAINSUBS;
	  }
	  else {
	      die "Unknown subsection $title for [main]";
	  }
      }
    }

    # check data
    $method or die "Data missing";

    # mark as read
    undef $DATA{"main"};

} # end if scan_main()


#############################################################################
#                                                                           #
#  Scan section [verbatim]                                                  #
#                                                                           #
#############################################################################

sub scan_verbatim {

    $DATA{"verbatim"} or return "";

    print "/*---------------------------------------------------------------------------*/\n";
    print "/* [verbatim] */\n\n";

    # There must be only one subsection: [start]

    # scan subsection 
    foreach my $subs (@{$DATA{"verbatim"}}) {
	my $title = ${$subs}{"title"};
	my $body = ${$subs}{"body"};
	if ($title eq "start") {
	    $body =~ s/(^|\n)\d+\:/\n/g;   # remove line info
	    print $body;
	}
	else {
	    die "No subsections allowed in [verbatim]";
	}
    }

    # mark as read
    undef $DATA{"verbatim"};
	
} # end if scan_verbatim()


#############################################################################
#                                                                           #
#  Scan section [validate]                                                  #
#                                                                           #
#############################################################################

sub scan_validate {

    $DATA{"validate"} or return "";

    $test_routines .= "test_validate();\n";

    print "/*---------------------------------------------------------------------------*/\n";
    print "/* [validate] */\n\n";

    my $section = "validate";
    my @generators;
    my @distributions;
    my $chi2;
    my $verifyhat;

    # scan subsection 
    foreach my $subs (@{$DATA{"validate"}}) {
	my $title = ${$subs}{"title"};
	my $body = ${$subs}{"body"};

      VALIDATESUBS: {
	  if ($title eq "start") { 
	      last VALIDATESUBS; }
	  if ($title eq "generators") {
	      @generators = get_blocks( $section, $body, 0 );
	      last VALIDATESUBS; }
	  if ($title eq "distributions") {
	      @distributions = get_blocks( $section, $body, 0 );
	      last VALIDATESUBS; }

	  # remove line info and empty lines
	  $body =~ s/(^|\n)\d+\:/\n/g;
	  $body =~ s/\n\s*\n/\n/g;
	  $body =~ s/^\s*\n//;

	  if ($title eq "test chi2") {
	      $chi2 = $body;
	      last VALIDATESUBS; }
	  if ($title eq "verify hat") {
	      $verifyhat = $body;
	      last VALIDATESUBS; }
	  else {
	      die "Unknown subsection $title for [$section]";
	  }
      }
    }

    # number of given distributions
    my $n_distributions = $#distributions + 1;

    # number of generators
    my $n_generators = $#generators + 1;

    print "/*---------------------------------------------------------------------------*/\n\n";
    print "/* [$section] */\n\n";
    print "void test_validate (void)\n{\n";

    print "\t/* suppress GCC compiler warning */\n";
    print "GCC_DIAG_OFF(unused-variable)\n\n";
    
    print "\tUNUR_DISTR *distr[".($n_distributions?$n_distributions:1)."];\n";
    print "\tUNUR_PAR *par;\n";
    print "\tUNUR_GEN *gen;\n";
    print "\tdouble *darray;\n";
    print "\tdouble fpm[10];\n";

    print "\tint n_tests_failed;\n";
    print "\tint rcode;\n";

    print "\n\t/* start test */\n";
    print "\tprintf(\"[validate \"); fflush(stdout);\n";
    print "\tfprintf(TESTLOG,\"\\n[validate]\\n\");\n\n";
    print "\t/* reset counter */\n\tn_tests_failed = 0;\n\n";

    print "\t/* set stop watch */\n";
    print "\tstopwatch_lap(&watch);\n\n";

    ## list of distributions ##

    print "\n/* distributions: $n_distributions */\n";
    foreach my $d (@distributions) {
	print "{\n$d}\n\n";
    }

    print "\t/* timing */\n";
    print "\tstopwatch_print(TESTLOG,\"\\n<*>setup time = %.3f ms\\n\", stopwatch_lap(&watch));\n\n";

    # --- chisquare GoF tests -----

    if ($chi2) {

	# analyse chi^2 tests
	my @chi2tests = split /\n/, $chi2;

	unless ($#chi2tests <= $#distributions) {
	    die "wrong number of chi2 tests ($#chi2tests > $#distributions)";
	}

	print "\tprintf(\"\\n(chi^2) \"); fflush(stdout);\n";
	print "\n/* chi^2 tests: ".($#generators+1)*$n_distributions." */\n\n";
	print "\tunur_set_default_debug(~UNUR_DEBUG_SAMPLE);\n";
	print "\tfprintf( TESTLOG,\"\\nChi^2 Test:\\n\");\n\n";
	
	foreach my $test (@chi2tests) {
	    die "invalide test line" unless ($test =~ /^\s*([fFxX]?)\s*<(\d+)>\s*(.+)/);
	    my $fullcheck_dist = $1;
	    my $n_distr = $2;
	    $test = $3;
	    $test =~ s/^\s+//;
	    $test =~ s/\s+$//;
	    my @gentest = split /\s+/, $test;
	    die "invalide number of test indicators" unless ($#gentest == $#generators);

	    print "/* distribution [$n_distr] */\n\n";

	    foreach (@generators) {
		# get entry for generator
		my $genline = $_;
		
		# remove [..] from par[..]
		# (it is just to number generators for convience)
		$genline =~ s/par\[(\d+)\]/par/g;
		
		# insert local copy of distribution object
		$genline =~ s/\@distr\@/distr_localcopy/g;
		
		# read what we have to test
		my $todo = shift @gentest;

		# test for an "fullcheckonly" flag
 		my $fullcheck_gen = 0;
		if ($todo =~ /^[fFxX](.+)/) {
		    $fullcheck_gen = 1;
		    $todo = $1;
		}

		# we encapsulate test into "if(TRUE)" or
                # "if(fullcheck)" statements.
		if ($fullcheck_dist || $fullcheck_gen) {
		    print "\tif(fullcheck) {\n";
		}
		else {
		    print "\tif(TRUE) {\n";
		}

		# nothing to do
		if ( $todo eq '.' ) {
		    print "\tprintf(\".\"); fflush(stdout);\n";
		    print "\t}\n\n";
		    next;
		}
		
		# split into lines again
		my @lines = split /\n/, $genline;
		
		# print lines 
		print "\tunur_reset_errno();\n";
		print "\tdo {\n";
		print "\tUNUR_DISTR *distr_localcopy = unur_distr_clone(distr[$n_distr]);\n";
		
		my $have_gen_lines = 0;
		foreach my $l (@lines) {
		    if ($l =~ /gen/ and !$have_gen_lines) {
			$have_gen_lines = 1;
			print "\tgen = unur_init(par);\n\tif (gen) {\n";
		    }
		    
		    print "$l\n";
		    
		    # we cannot run the test if par == NULL.
		    # then essential parameters are missing.
		    # if ($l =~ /^\s*par\s*=\s*/) {
		    # print "\tif (par==NULL) { printf(\"X\"); fflush(stdout); break; }\n";
		    #}
		}
		
		if ($have_gen_lines) {
		    print "\t}\n";
		}
		else {
		    print "\tgen = unur_init(par);\n";
		}
		print "\trcode = run_validate_chi2(TESTLOG,0,gen,distr[$n_distr],'$todo');\n";
		print "\tn_tests_failed += (rcode==UNUR_SUCCESS)?0:1;\n";
		print "\tn_tests_failed += (rcode==UNUR_FAILURE)?1000:0;\n";
		print "\tunur_free(gen);\n";
		print "\tunur_distr_free(distr_localcopy);\n";
		print "\t} while (0);\n";

		# end of if() statement
		print "\t}\n\n";
	    }	    
	}

	print "\t/* timing */\n";
	print "\tstopwatch_print(TESTLOG,\"\\n<*>time = %.0f ms\\n\", stopwatch_lap(&watch));\n\n";
    }

    ## run in verify mode  ##

    if ($verifyhat) {

	# analyse verify hat tests
	my @verifyhattests = split /\n/, $verifyhat; 
	die "wrong number of verify hat tests" unless ($#verifyhattests <= $#distributions);

	print "\tprintf(\"\\n(verify hat) \"); fflush(stdout);\n";
	print "\n/* verify hat tests: ".($#generators+1)*$n_distributions." */\n\n";
	print "\tunur_set_default_debug(~UNUR_DEBUG_SAMPLE);\n";
	print "\tfprintf( TESTLOG,\"\\nVerify Hat Test (squeeze <= PDF <= hat):\\n\");\n\n";
	
	foreach my $test (@verifyhattests) {
	    die "invalide test line" unless ($test =~ /^\s*([fFxX]?)\s*<(\d+)>\s*(.+)/);
	    my $fullcheck_dist = $1;
	    my $n_distr = $2;
	    $test = $3;
	    $test =~ s/^\s+//;
	    $test =~ s/\s+$//;
	    my @gentest = split /\s+/, $test;
	    die "invalide number of test indicators" unless ($#gentest == $#generators);

	    print "/* distribution [$n_distr] */\n\n";

	    foreach (@generators) {
		# get entry for generator
		my $genline = $_;
		
		# remove [..] from par[..]
		# (it is just to number generators for convience)
		$genline =~ s/par\[(\d+)\]/par/g;
		
		# insert local copy of distribution object
		$genline =~ s/\@distr\@/distr_localcopy/g;
		
		# read what we have to test
		my $todo = shift @gentest;

		# replace '+' by '~'
		$todo =~ s/\+/\~/;

		# test for an "fullcheckonly" flag
 		my $fullcheck_gen = 0;
		if ($todo =~ /^[fFxX](.+)/) {
		    $fullcheck_gen = 1;
		    $todo = $1;
		}

		# we encapsulate test into "if(TRUE)" or
                # "if(fullcheck)" statements.
		if ($fullcheck_dist || $fullcheck_gen) {
		    print "\tif(fullcheck) {\n";
		}
		else {
		    print "\tif(TRUE) {\n";
		}

		# nothing to do
		if ( $todo eq '.' ) {
		    print "\tprintf(\".\"); fflush(stdout);\n";
		    print "\t}\n\n";
		    next;
		}
		
		# split into lines again
		my @lines = split /\n/, $genline;
		
		# print lines 
		print "\tunur_reset_errno();\n";
		print "\tdo {\n";
		print "\tUNUR_DISTR *distr_localcopy = unur_distr_clone(distr[$n_distr]);\n";
		
		my $have_gen_lines = 0;
		foreach my $l (@lines) {
		    if ($l =~ /gen/ and !$have_gen_lines) {
			$have_gen_lines = 1;
			print "\tgen = unur_init(par);\n\tif (gen) {\n";
		    }
		    
		    print "$l\n";
		    if ($l =~ /^\s*par\s*=/) {
			print "\tunur_$method\_set_pedantic(par,0);\n";

			# we cannot run the test if par == NULL.
			# then essential parameters are missing.
			# print "\tif (par==NULL) { printf(\"X\"); fflush(stdout); break; }\n";
		    }

		}
		
		if ($have_gen_lines) {
		    print "\t}\n";
		}
		else {
		    print "\tgen = unur_init(par);\n";
		}
		print "\tif (gen) unur_$method\_chg_verify(gen,1);\n";
		# such an error is fatal. so we must make sure that we get a FAIL
		# and override the sloppy definition of FAIL (CHI2_FAILURES_TOLERATED are allowed for chi^2)
		print "\tn_tests_failed += (run_validate_verifyhat(TESTLOG,0,gen,distr[$n_distr],'$todo')==UNUR_SUCCESS)?0:1000;\n";
		print "\tunur_free(gen);\n\n";
		print "\tunur_distr_free(distr_localcopy);\n";
		print "\t} while (0);\n";

		# end of if() statement
		print "\t}\n\n";
	    }	    
	}

	print "\t/* timing */\n";
	print "\tstopwatch_print(TESTLOG,\"\\n<*>time = %.0f ms\\n\", stopwatch_lap(&watch));\n\n";
    }

    ## free distributions ##

    print "\n/* free distributions */\n";
    for (my $i=0; $i<$n_distributions; ++$i) {
	print "\tunur_distr_free(distr[$i]);\n";
    }

    ## end ##

    print "\n\t/* test finished */\n";
    print "\ttest_ok &= (n_tests_failed>CHI2_FAILURES_TOLERATED) ? 0 : 1;\n";
    print "\t/* we accept CHI2_FAILURES_TOLERATED failures */\n";
    print "\t(n_tests_failed>CHI2_FAILURES_TOLERATED) ? printf(\" ==> failed] \") : printf(\" ==> ok] \");\n";

    print "\n} /* end of test_validate */\n\n";
    print "/* enable GCC compiler warning */\n";
    print "GCC_DIAG_ON(unused-variable)\n\n";

    # mark as read
    undef $DATA{"validate"};

} # end if scan_validate()



#############################################################################
#                                                                           #
#  Scan section [special]                                                   #
#                                                                           #
#############################################################################

sub scan_special {

    my $section = "special";

    $DATA{"$section"} or return "";

    $test_routines .= "test_$section();\n";

    print <<EOM;
/*---------------------------------------------------------------------------*/
/* [$section] */

void test_$section (void)
{
	/* set boolean to FALSE */
	int FAILED = 0;

EOM

    # There must be only one subsection: [start]

    # scan subsection 
    foreach my $subs (@{$DATA{$section}}) {
	my $title = ${$subs}{"title"};
	my $body = ${$subs}{"body"};
	if ($title eq "decl") {
	    $body =~ s/(^|\n)\d+\:/\n/g;   # remove line info
	    print $body;
	    print 
		"/* start test */\n".
		"printf(\"[$section \"); fflush(stdout);\n".
		"fprintf(TESTLOG,\"\\n[$section]\\n\");\n\n";
	    print
		"/* set stop watch */\n".
		"stopwatch_lap(&watch);\n\n";
  	}
	elsif ($title eq "start") {
	    $body =~ s/(^|\n)\d+\:/\n/g;   # remove line info
	    print $body;
	}
	else {
	    die "No subsections allowed in [$section]";
	}
    }


  # print section closing ...
  print <<EOM;

	/* timing */
	stopwatch_print(TESTLOG,"\\n<*>time = %.3f ms\\n\\n", stopwatch_lap(&watch));

	/* test finished */
	test_ok &= (FAILED) ? 0 : 1;
	(FAILED) ? printf(" ==> failed] ") : printf(" ==> ok] ");

} /* end of test_$section() */

EOM

    # mark as read
    undef $DATA{$section};

} # end if scan_special()


#############################################################################
#                                                                           #
#  Scan all other sections                                                  #
#                                                                           #
#############################################################################

sub scan_other {
    my $section = shift;

    $DATA{$section} or return "";

    $test_routines .= "test_$section();\n";

    print <<EOM;
/*---------------------------------------------------------------------------*/
/* [$section] */

void test_$section (void)
{
        int n_tests_failed;          /* number of failed tests */

	/* start test */
	printf("[$section "); fflush(stdout);
	fprintf(TESTLOG,"\\n[$section]\\n");

	/* reset counter */
	n_tests_failed = 0;

	/* set stop watch */
	stopwatch_lap(&watch);
  
EOM

    # scan subsection 
    foreach my $subs (@{$DATA{$section}}) {
	my $title = ${$subs}{"title"};
	my $header = ${$subs}{"header"};
	my $body = ${$subs}{"body"};

	next if $title eq "start";

	# clear variables ...
	my $subsection_closing = "";

	print "{ /* $title */\n";

	# parse header
	my @lines = split /\n/, $header;
	my $decl;   # contains declarations
	my $code;   # contains code
	foreach my $line (@lines) {
	    $line =~ s/^(\d+)\://;
	    my $lineno = $1  or die "internal error";
	    if ($line =~ /\s+gen\s+=/) { 
		$decl .= "UNUR_GEN   *gen = NULL\;\n";
#		$line = "UNUR_GEN   *gen = NULL\;\n".$line;
		$subsection_closing .= "unur_free(gen)\;\n";
	    }
	    if ($line =~ /\s+par\s+=/) {
		$decl .= "UNUR_PAR   *par = NULL\;\n"; 
	    }
	    if ($line =~ /\s+distr\s+=/) {
		$decl .= "UNUR_DISTR *distr = NULL\;\n"; 
#		$line = "UNUR_DISTR *distr = NULL\;\n".$line; 
		$subsection_closing .= "unur_distr_free(distr)\;\n";
	    }
	    $code .= "$line\n";
	    # lines indicated with "<-- ! NULL" must not produce a NULL pointer 
	    $code =~ s/^(.*)=(.*)<--\s+!\s*NULL\s*\n/$1=$2\nabort_if_NULL\(TESTLOG, $lineno, $1\)\;\n/mg;
	}
	print $decl, $code, "\n";

	# parse body
	my @blocks = get_blocks( $section, $body, 1 );
	foreach my $block (@blocks) {
	    $block =~ s/^(\d+)\://;
	    my $lineno = $1  or die "internal error";

	    # analyze string ...
	    my ($code,$test_command,$errno) = split /-->/, $block;

	    # is there any test ?
	    unless ($test_command) {
		print $code if defined $code;
		next;
	    }

	    # split code into lines again ...
	    my @blines = split /\n/, $code;

	    # get last C command ...
	    undef my $last_C_line;
	    while ( $last_C_line = pop @blines ) {
		last if $last_C_line =~ /\w+/;
		last if !defined $last_C_line;
	    }
	    $last_C_line =~ s/;\s*$// if defined $last_C_line;
	    
	    # print ...
	    print "\nunur_reset_errno();\n";
	    
	    # C lines ...
	    foreach my $bl (@blines) {
		print "$bl\n";
	    }
	    
	    # test ...
	    print_test_command( $test_command, $last_C_line, $lineno );
	    
	    # error code ...
	    $errno =~ s/[\s\n]+//g if defined $errno;
	    print "n_tests_failed += (check_errorcode(TESTLOG,$lineno,$errno)==UNUR_SUCCESS)?0:1\;\n" if $errno;
	}

	# close subsection ...
	print "$subsection_closing}\n\n";
    }

  # print section closing ...
  print <<EOM;

	/* timing */
	stopwatch_print(TESTLOG,"\\n<*>time = %.3f ms\\n\\n", stopwatch_lap(&watch));

	/* test finished */
	test_ok &= (n_tests_failed) ? 0 : 1;
	(n_tests_failed) ? printf(" ==> failed] ") : printf(" ==> ok] ");

} /* end of test_$section() */

EOM

    # mark as read
    undef $DATA{$section};

} # end if scan_other()


#############################################################################
#                                                                           #
#  break into blocks (separated by blank lines)                             #
#                                                                           #
#############################################################################

sub get_blocks {
    my $section = shift;
    my $body = shift;
    my $withlineno = shift;

    my @blocks;
    my $block;
    my @lines = split /\n/, $body;

    foreach my $line (@lines) {
	$line .= "\n";
	$line =~ s/^(\d+)\://;
	my $lineno = $1  or die "internal error";

	# convert line starting with ~ ...
	$line =~ s/~/unur_$method\_$section/ if $line =~ /^~/;

	if ($line =~ /\w+/) { 
	    # append line to block
	    $block .= $line; }
	else {
	    # empty line --> block completed
	    # (we ignore lines without letters)
	    $lineno = $withlineno ? "$lineno:" : "";
	    push @blocks, "$lineno$block" if $block =~ /\w+/;

	    # goto next block
	    $block = "";
	}
    } 
    
    return @blocks;
} # end of get_blocks()


#############################################################################
#                                                                           #
#  print test command                                                       #
#                                                                           #
#############################################################################

sub print_test_command {
    my $test_command = $_[0];
    my $last_C_line = $_[1];
    my $lineno = $_[2];

  SWITCH: {
      if ($test_command =~ /^\s*none\s*/ ) {
	  $test_command =~ s/\s+//g;
	  print "$last_C_line\;\n";
	  last SWITCH;
      }
      if ($test_command =~ /^\s*expected_NULL\s*/ or 
	  $test_command =~ /^\s*expected_setfailed\s*/ or 
	  $test_command =~ /^\s*expected_zero\s*/ or 
	  $test_command =~ /^\s*expected_INFINITY\s*/ or 
	  $test_command =~ /^\s*expected_negINFINITY\s*/ or 
	  $test_command =~ /^\s*expected_INTMAX\s*/ or 
	  $test_command =~ /^\s*expected_reinit\s*/) {
	  $test_command =~ s/\s+//g;
	  print "n_tests_failed += (check_$test_command\(TESTLOG,$lineno,($last_C_line))==UNUR_SUCCESS)?0:1\;\n";
	  last SWITCH;
      }
      if ($test_command =~ /^\s*compare_sequence_gen\s*$/ or
	  $test_command =~ /^\s*compare_sequence_gen_start\s*$/ ) {
	  $test_command =~ s/\s+//g;
	  print "$last_C_line\;\n";
	  print "n_tests_failed += ($test_command\(TESTLOG,$lineno,gen,COMPARE_SAMPLE_SIZE)==UNUR_SUCCESS)?0:1;\n";
	  last SWITCH;
      }
      if ($test_command =~ /^\s*compare_sequence_par\s*$/ or
	  $test_command =~ /^\s*compare_sequence_par_start\s*$/ ) {
	  $test_command =~ s/\s+//g;
	  print "$last_C_line\;\n";
	  print "n_tests_failed += ($test_command\(TESTLOG,$lineno,par,COMPARE_SAMPLE_SIZE)==UNUR_SUCCESS)?0:1;\n";
	  last SWITCH;
      }
      if ($test_command =~ /^\s*compare_sequence_urng_start\s*$/ ) {
	  $test_command =~ s/\s+//g;
	  print "$last_C_line\;\n";
	  print "$test_command\(TESTLOG,$lineno,COMPARE_SAMPLE_SIZE);\n";
	  last SWITCH;
      }
      if ($test_command =~ /^\s*run_verify_generator\s*$/) {
	  print "$last_C_line\;\n";
	  print "$test_command(TESTLOG,$lineno,par);\n";
	  last SWITCH;
      }

      # otherwise
      die "Unknown test command: \"$test_command\"";
  }

} # end print_test_command() 

#############################################################################
#                                                                           #
#  print C header, main, and auxiliary routines                             #
#                                                                           #
#############################################################################

sub print_C_header {

    print "/*\n\tfile automatically generated by $name_program\n\t";
    print scalar localtime;
    print "\n*/\n\n";

    print <<EOM;
/*****************************************************************************
 *                                                                           *
 *          UNU.RAN -- Universal Non-Uniform Random number generator         *
 *                                                                           *
 *****************************************************************************/
    
/**
 ** Tests for $METHOD
 **/
    
/*---------------------------------------------------------------------------*/
#include "testunuran.h"

#ifdef UNUR_URNG_DEFAULT_RNGSTREAM
#include <RngStream.h>
#endif
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/* global variables                                                          */

static FILE *TESTLOG;               /* test log file                         */
static FILE *UNURANLOG;             /* unuran log file                       */

static int test_ok = TRUE;          /* all tests ok (boolean)                */
static int fullcheck = FALSE;       /* whether all checks are performed      */ 

static TIMER watch;                 /* stop watch                            */

/*---------------------------------------------------------------------------*/

void run_verify_generator( FILE *LOG, int line, UNUR_PAR *par );

$verify_prototype

/*---------------------------------------------------------------------------*/

void test_new (void);
void test_set (void);
void test_get (void);
void test_chg (void);
void test_init (void);
void test_reinit (void);
void test_sample (void);
void test_validate (void);
void test_special(void);

/*---------------------------------------------------------------------------*/

$C_header_aux

/*---------------------------------------------------------------------------*/

#ifndef CHI2_FAILURES_TOLERATED
#  define CHI2_FAILURES_TOLERATED DEFAULT_CHI2_FAILURES_TOLERATED
#endif

EOM

} # end of print_C_header()

#############################################################################

sub print_C_main {

    print <<EOM;

/*---------------------------------------------------------------------------*/

int main(void)
{ 
        unsigned long seed;
	char *str_seed, *str_tail;

	/* start stop watch */
	stopwatch_init();
	stopwatch_start(&watch);

        /* open log file for unuran and set output stream for unuran messages */
        UNURANLOG = fopen( "$file_unuranlog","w" );
        abort_if_NULL( stderr,-1, UNURANLOG );
        unur_set_stream( UNURANLOG );

        /* open log file for testing */
	TESTLOG = fopen( "$file_testlog","w" );
	abort_if_NULL( stderr,-1, TESTLOG );

        /* seed for uniform generators */

	/* seed set by environment */
	str_seed = getenv("SEED");

	if (str_seed != NULL) {
	    seed = strtol(str_seed, &str_tail, 10);
	    if (seed == 0u) 
		seed = $seed;
	}
	else {
#ifdef SEED
	    seed = SEED;
#else
	    seed = $seed;
#endif
	}

        /* seed build-in uniform generators */
        unur_urng_MRG31k3p_seed(NULL,seed);
        unur_urng_fish_seed(NULL,seed);
	unur_urng_mstd_seed(NULL,seed);

	/* seed uniform random number generator */
#ifdef UNUR_URNG_UNURAN
#  ifdef UNUR_URNG_DEFAULT_RNGSTREAM
	{
	        unsigned long sa[6];
	        int i;
	        for (i=0; i<6; i++) sa[i] = seed;
                RngStream_SetPackageSeed(sa);
        }
#  else
	if (unur_urng_seed(NULL,seed) != UNUR_SUCCESS) {
	        fprintf(stderr,"WARNING: Seed could not be set at random\\n");
                seed = ~0u;
	}
#  endif  /* UNUR_URNG_DEFAULT_RNGSTREAM */
#endif  /* UNUR_URNG_UNURAN */
 
	/* set default debugging flag */
	unur_set_default_debug(UNUR_DEBUG_ALL);

        /* detect required check mode */
        fullcheck = (getenv("UNURANFULLCHECK")==NULL) ? FALSE : TRUE;

	/* write header into log file */
        print_test_log_header( TESTLOG, seed, fullcheck );

	/* set timer for sending SIGALRM signal */
	set_alarm(TESTLOG);

	/* start test */
	printf("$method: ");

	/* run tests */
$test_routines

	/* test finished */
	printf("\\n");  fflush(stdout);

	/* close log files */
	fprintf(TESTLOG,"\\n====================================================\\n\\n");
	if (test_ok)
		fprintf(TESTLOG,"All tests PASSED.\\n");
	else
		fprintf(TESTLOG,"Test(s) FAILED.\\n");

	/* timing */
	stopwatch_print(TESTLOG,"\\n<*>total time = %.0f ms\\n\\n", stopwatch_stop(&watch));

	fclose(UNURANLOG);
	fclose(TESTLOG);

	/* free memory */
	compare_free_memory();
	unur_urng_free(unur_get_default_urng());
	unur_urng_free(unur_get_default_urng_aux());

	/* exit */
	exit( (test_ok) ? EXIT_SUCCESS : EXIT_FAILURE );

} /* end of main */

EOM

} # end of print_C_main()

#############################################################################

sub print_C_routines {

    print <<EOM;

/*---------------------------------------------------------------------------*/
/* run generator in verifying mode */

void run_verify_generator( FILE *LOG, int line, UNUR_PAR *par )
{
	UNUR_GEN *gen;
	int i;

	/* switch to verifying mode */
	unur_$method\_set_verify(par,1);

	/* initialize generator */
	gen = unur_init( par ); abort_if_NULL(LOG, line, gen);

	/* run generator */
	for (i=0; i<VIOLATE_SAMPLE_SIZE; i++)
		unur_sample_cont(gen);

	/* destroy generator */
	unur_free(gen); 

} /* end of run_verify_generator() */

EOM

} # end of print_C_routines()

#############################################################################

sub make_unur_set_verify_routine {
    # add a dummy routine when there is no verify routine for the method

    my $verify = "unur_$method\_set_verify";
    undef my $have_found;

    foreach my $h (@header_files) {
	open (H, $h) or next;
	while (<H>) {
	    $have_found = 1 if /$verify/;
	}
	close (H);
    }

    unless ($have_found) {
	$verify_routine = "int $verify(UNUR_PAR *par ATTRIBUTE__UNUSED, int verify ATTRIBUTE__UNUSED) {return 0;}\n";
	$verify_prototype = "int unur_$method\_set_verify( UNUR_PAR *par, int verify);\n";

    }
    else {
	$verify_routine = "";
	$verify_prototype = "";
    }
}

#############################################################################
