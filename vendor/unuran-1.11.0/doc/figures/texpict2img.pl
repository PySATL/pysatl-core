#!/usr/bin/perl
############################################################

use strict;
use Getopt::Std;

############################################################
# programms

my $LATEX="latex";
my $DVIPS="dvips";
my $GS="gs";
my $PS2EPSI="ps2epsi";
my $EPSTOPDF="epstopdf";

############################################################

sub usage {
    my $progname = $0;
    $progname =~ s#^.*/##g;
        
    print STDERR <<EOM;
usage: $progname <file.tex> 
      
   Read TeX file that contains a single picture and
   converts it into EPS, PDF, PNG, JPEG and TXT format.
   It is assumed that these pictures are produced by pstricks 
   commands.
   Each of these is written into the respective file <file.FORMAT>
   where FORMAT is "eps", "pdf", "png", "jpg" or "txt".
   The path information is stripped and the files are written in
   working directory (i.e. the directory where the command 
   has been executed).

   Options:
   
   -r <num> ... resolution for PNG and JPEG format [default: 150]

   -h <num> ... height and width for TXT format (in characters)
   -w <num>       [defautl: -h 30 - w 77]

   Height and width in PS and PDF format are given by units in TeX file.

   WARNING! QUICK & DIRTY HACK!
EOM
    exit -1;
}

############################################################
# Defaults

my $resolution = 150;
my $txt_isize = 77;
my $txt_jsize = 30;

############################################################
# Read arguments

our ($opt_r, $opt_h, $opt_w);
getopts('r:h:w:');
$resolution = $opt_r if $opt_r;
$txt_isize = $opt_w if $opt_w;
$txt_jsize = $opt_h if $opt_h;

print STDERR "resolution = $resolution\n"; 
print STDERR "TXT width  = $txt_isize\n"; 
print STDERR "TXT height = $txt_jsize\n"; 
print STDERR "----------------------\n";

############################################################
# Ret name of TeX file 
# and compose names of all temp and output files
#
# Read name of TeX file from argument list ...
my $tex_file = shift;
(usage and die) unless $tex_file;

# Get base of TeX file name
my $file_base = $tex_file;
$file_base =~ s[^.*/]{};
$file_base =~ s/(.*)\..*$/$1/;

# Output files 
my $eps_file = "$file_base.eps";
my $pdf_file = "$file_base.pdf";
my $png_file = "$file_base.png";
my $jpg_file = "$file_base.jpg";
my $txt_file = "$file_base.txt";

print STDERR "TeX file = $tex_file\n";
print STDERR "EPS file = $eps_file\n";
print STDERR "PDF file = $pdf_file\n";
print STDERR "PNG file = $png_file\n";
print STDERR "JPG file = $jpg_file\n";
print STDERR "TXT file = $txt_file\n";
print STDERR "----------------------\n";

# Temp files  
my $tmp_base = "tmp-$file_base-$$";
my $tmp_tex_file = "$tmp_base.tex";     # TeX file for processing picture
my $tmp_dvi_file = "$tmp_base.dvi";     # dvi file with picture
my $tmp_ps_file_1 = "$tmp_base-1.ps";   # result of dvips
my $tmp_ps_file_2 = "$tmp_base-2.ps";   # stripped PS file (with wrong bounding boxes)
my $tmp_ps_file = "$tmp_base.ps";       # PS file with correct bounding boxes
my $tmp_epsi_file = "$tmp_base.epsi";   # result of ps2epsi (to be needed for epstopdf)

############################################################

# Read TeX file and estimate TeX bounding box
my $tex_bx0 = "a";
my $tex_by0 = 0;
my $tex_bx1 = 0;
my $tex_by1 = 0;
my $tex_code;      

open TEX, "$tex_file";
while (<TEX>) {
    # store tex code
    $tex_code .= $_;

    if ( /\\begin\{pspicture\}\((-?[\d\.]+),\s*(-?[\d\.]+)\)\((-?[\d\.]+),\s*(-?[\d\.]+)\)/ ) {
	print STDERR "junk!!\n\n";
	$tex_bx0 = $1; $tex_by0 = $2;
	$tex_bx1 = $3; $tex_by1 = $4;
    }
}
close TEX;

if ($tex_bx0 eq "a") { die "Cannot find bounding box"; }

print STDERR "TeX bounding box = ($tex_bx0,$tex_by0) x ($tex_bx1,$tex_by1)\n";
print STDERR "----------------------\n";

############################################################

print STDERR "make temp file $tmp_tex_file\n";

open TEX, ">$tmp_tex_file" or die "Cannot open file $tmp_tex_file";
print TEX
    "\\documentclass{article}\n",
    "\\usepackage{pstricks,pst-plot,pst-node}\n",
    "\\usepackage{amsfonts}\n",
    "\\usepackage{amssymb}\n",
    "\\pagestyle{empty}\n",
    "\\parindent 0pt\n",
    "\\parskip 0pt\n",
    "\\newcommand{\\onehalf}{{\\textstyle\\frac{1}{2}}}\n",
    "\\begin{document}\n",
    "\\mbox{}\\vfill\n",
    "\\centering\n",
    "\\input{$tex_file}\n",
    "\\vfill\\mbox{}\n",
    "\\end{document}\n";
close TEX;

# Make dvi
system("$LATEX $tmp_tex_file");
print STDERR "----------------------\n";

# Make ps
system("$DVIPS -o $tmp_ps_file_1 $tmp_dvi_file");
print STDERR "----------------------\n";

# Get PS bounding box
my $bbox = `$GS -q -sDEVICE=bbox -dNOPAUSE -dBATCH $tmp_ps_file_1 2>&1`;
my $boundingbox;
my $hiresboundingbox;
foreach my $l (split /\n/, $bbox) {
    $boundingbox = $l if ($l =~ /%%BoundingBox:/);
    $hiresboundingbox = $l if ($l =~ /%%HiResBoundingBox:/);
}
($boundingbox) or die "Cannot find bounding box";
$boundingbox =~ m/^%%BoundingBox: ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)/m;
my $PS_bx0 = $1;
my $PS_by0 = $2;
my $PS_bx1 = $3;
my $PS_by1 = $4;
system("cat $tmp_ps_file_1 | sed -e 's/%%BoundingBox:.*\$/$boundingbox/' | sed -e 's/%%HiResBoundingBox:.*\$/$hiresboundingbox/'> $tmp_ps_file");

############################################################

# Make eps
print STDERR "Create EPS file ...\n";
system("$GS -q -dNOPAUSE -dBATCH -sDEVICE=pswrite -sOutputFile=$tmp_ps_file_2 -f $tmp_ps_file_1");
system("cat $tmp_ps_file_2 | sed -e 's/%%BoundingBox:.*\$/$boundingbox/' | sed -e 's/%%HiResBoundingBox:.*\$/$hiresboundingbox/'> $eps_file");

# Make pdf
print STDERR "Create PDF file ...\n";
system("$PS2EPSI $tmp_ps_file_1 $tmp_epsi_file");
system("$EPSTOPDF --outfile=$pdf_file $tmp_epsi_file");

# Make png
print STDERR "Create PNG file ...\n";
my $translate="-$PS_bx0 -$PS_by0 translate";
my $geometry = int($resolution*($PS_bx1-$PS_bx0)/72) ."x". int($resolution*($PS_by1-$PS_by0)/72);
system("$GS -q -dNOPAUSE -dBATCH -sDEVICE=pnggray -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=$png_file -r$resolution -g$geometry -c $translate -f $tmp_ps_file");

# Make jpeg
print STDERR "Create JPEG file ...\n";
system("$GS -q -dNOPAUSE -dBATCH -sDEVICE=jpeggray -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=$jpg_file -r$resolution -g$geometry -c $translate -f $tmp_ps_file");
print STDERR "----------------------\n";

############################################################

# Make txt file
# (very primitive support)

print STDERR "Create TXT file ...\n";

# make blank page
my $txt_string = " " x ($txt_isize);
$txt_string .= "\n";
$txt_string = $txt_string x ($txt_jsize);

# remove % from code
$tex_code =~ s/\%//g;

# scan tex code
while (1) {
    # axes
    if ($tex_code =~ /\\psaxes\s*(\[.*?\])?(\{.*?\})?(.*?)([\n])/s) {
	my $line = $3;
	$line =~ s/^\s*\(//;
	$line =~ s/\)\s*$//;
	$tex_code =~ s/\\psaxes\s*(\[.*?\])?(\{.*?\})?(.*?)([\n])/$4/s;

	my @points = split /\s*\)\s*\(\s*/, $line;
	if ($#points == 0) {
	    (my $x, my $y) = split /\,/, $points[0], 2;
	    txt_joint_points("0,0","$x,0");
	    txt_joint_points("0,0","0,$y");
	    txt_draw_point($x,0,">");
	    txt_draw_point(0,$y,"^");
	}
	else {
	    die "Cannot handle psaxes: #points = $#points; tex = $line!\n\n\n";
	}
	next;
    }
	
    # lines 
    if ($tex_code =~ /\\psline\s*(\[.*?\])?(\{.*?\})?(.*?)([\n])/s) {
	my $line = $3;
	$line =~ s/^\s*\(//;
	$line =~ s/\)\s*$//;
	$tex_code =~ s/\\psline\s*(\[.*?\])?(\{.*?\})?(.*?)([\n])/$3/s;

	my @points = split /\s*\)\s*\(\s*/, $line;
	for my $n (0..$#points-1) {
	    txt_joint_points($points[$n],$points[$n+1]);
	}
	next;
    }

    # circles
    if ($tex_code =~ /\\pscircle\*?\s*(\[.*?\])?(.*?)(\{.*?\})([\n])/s) {
	my $line = $2;
	$line =~ s/^\s*\(//;
	$line =~ s/\)\s*$//;
	$tex_code =~ s/\\pscircle\*?\s*(\[.*?\])?(.*?)(\{.*?\})([\n])/$4/s;
	(my $x, my $y) = split /\,/, $line, 2;
	txt_draw_point($x,$y,"o");
	next;
    }

    # plot
    if ($tex_code =~ /\\psplot\s*(\[.*?\])?\s*\{(.*?)\}\s*\{(.*?)\}\s*\{(.*?)\}\s*([\n])/s) {
	my $x0 = $2;
	my $x1 = $3;
	my $code = $4;
	print "plot [$x0:$x1] $code\n";
	$tex_code =~ s/\\psplot\s*(\[.*?\])?\s*\{(.*?)\}\s*\{(.*?)\}\s*\{(.*?)\}\s*([\n])/$5/s;
	txt_plot($x0,$x1,$code);
	next;
    }

    # parametric plot
    if ($tex_code =~ /\\parametricplot\s*(\[.*?\])?\s*\{(.*?)\}\s*\{(.*?)\}\s*\{(.*?)\}\s*([\n])/s) {
	my $x0 = $2;
	my $x1 = $3;
	my $code = $4;
	print "parametric plot [$x0:$x1] $code\n";
	$tex_code =~ s/\\parametricplot\s*(\[.*?\])?\s*\{(.*?)\}\s*\{(.*?)\}\s*\{(.*?)\}\s*([\n])/$5/s;
	txt_parametricplot($x0,$x1,$code);
	next;
    }
  
    else { 
	last;
    }
}

# print page
open TXT, ">$txt_file";
print TXT $txt_string;
close TXT;

print STDERR "----------------------\n";

############################################################
# Clear working space

system("rm -v $tmp_base*.*");

exit 0;

############################################################

sub txt_draw {
    (my $i, my $j, my $c) = @_;

    $j = ($txt_jsize-1)-$j;

    $i=0 if $i<0;
    $i=$txt_isize-1 if $i>=$txt_isize;
    $j=0 if $j<0;
    $j=$txt_jsize-1 if $j>=$txt_jsize;

    
    my $pos = $j*($txt_isize+1) + $i;
    my $oc = substr($txt_string, $pos, 1); 

  CHAR: {
      if ($c eq "-" and $oc eq "|" || $oc eq "+") { $c="+"; last CHAR; }
      if ($c eq "|" and $oc eq "-" || $oc eq "+") { $c="+"; last CHAR; }
  }

    substr($txt_string, $pos, 1) = $c; 
}

sub txt_draw_point {
    (my $x, my $y, my $c) = @_;

    (my $i, my $j) = txt_xy2ij($x,$y);
    txt_draw($i,$j,$c);
}

sub txt_xy2ij {
    (my $x, my $y) = @_;

    my $i = int( ($x-$tex_bx0)/($tex_bx1-$tex_bx0) * $txt_isize);
    my $j = int( ($y-$tex_by0)/($tex_by1-$tex_by0) * $txt_jsize);

    return ($i,$j);
}

sub txt_joint_points {
    (my $first, my $second) = @_;

    (my $x0, my $y0) = split /\,/, $first, 2;
    (my $x1, my $y1) = split /\,/, $second, 2;

    print STDERR "line ($x0,$y0) -> ($x1,$y1): ";

    # get drawing character (dependent on slope)
    my $c = ".";
    (my $i0, my $j0) = txt_xy2ij($x0,$y0);
    (my $i1, my $j1) = txt_xy2ij($x1,$y1);
    my $s = atan2 ($j1-$j0,$i1-$i0);
  CHAR: {
      if ($s > -0.16 && $s < 0.16 or $s > 3.00 or $s < -3.00) { $c="-"; last CHAR; }
      if ($s > 1.41 && $s < 1.73 or $s > -1.73 && $s < -1.41) { $c="|"; last CHAR; }
      if ($s > 0.62 && $s < 0.95 or $s > -2.52 && $s < -2.28) { $c="/"; last CHAR; }
      if ($s < -0.62 && $s > -0.95 or $s < 2.52 && $s > 2.28) { $c="\\"; last CHAR; }
  }

    # number of steps 
    my $steps = 1;
    if ($s > 0.78 && $s < 2.36 or $s < -0.78 && $s > -2.36) {
	$steps = $j1-$j0;
    }
    else {
	$steps = $i1-$i0;
    }
    $steps *= -1 if $steps < 0;

    print STDERR "slope = $s, steps = $steps, char = $c\n";

    # draw line
    for my $t (0..$steps) {
	my $x = ($t * $x0 + ($steps-$t)*$x1)/$steps;
	my $y = ($t * $y0 + ($steps-$t)*$y1)/$steps;
	txt_draw_point($x,$y,$c);
    }
}

sub txt_plot {
    (my $x0, my $x1, my $code) = @_;

    # trim blanks
    $code =~ s/^\s+//;
    $code =~ s/\s+$//;

    # number of steps 
    (my $i0, my $j0) = txt_xy2ij($x0,"0");
    (my $i1, my $j1) = txt_xy2ij($x1,"0");
    my $steps = $i1-$i0;
    $steps *= -1 if $steps < 0;

    # make plot
    for my $t (0..$steps) {
	my $x = ($t * $x0 + ($steps-$t)*$x1)/$steps;
	my $y = (txt_plot_eval($x,$code))[0];
	txt_draw_point($x,$y,"*");
    }
}

sub txt_parametricplot {
    (my $t0, my $t1, my $code) = @_;

    # trim blanks
    $code =~ s/^\s+//;
    $code =~ s/\s+$//;

    # number of steps 
    my $steps = 100;

    # make plot
    for my $s (0..$steps) {
	my $t = ($s * $t0 + ($steps-$s)*$t1)/$steps;
	my @x = txt_plot_eval($t,$code);
	txt_draw_point($x[0],$x[1],"*");
    }
}

#  \parametricplot[plotstyle=curve,linestyle=none,fillstyle=solid,fillcolor=gray]%
#  {0}{180}{t cos t sin 0.75 mul}

sub txt_plot_eval {
    (my $x, my $code) = @_;

    my @commands = split /\s+/, $code;
    my @stack;

    foreach my $cmd (@commands) {
      CMD:{
	  if ($cmd =~ /^\-?\d*\.?\d*$/) { 
	      push @stack, $cmd;
	      last CMD; }
	  if ($cmd eq "x" or $cmd eq "t") { 
	      push @stack, $x;
	      last CMD; }
	  if ($cmd eq "neg") { 
	      my $a = pop @stack;
	      push @stack, (-$a);
	      last CMD; }
	  if ($cmd eq "mul") { 
	      my $a = pop @stack;
	      my $b = pop @stack;
	      push @stack, ($a*$b);
	      last CMD; }
	  if ($cmd eq "sub") { 
	      my $a = pop @stack;
	      my $b = pop @stack;
	      push @stack, ($b-$a);
	      last CMD; }
	  if ($cmd eq "exp") { 
	      my $a = pop @stack;
	      my $b = pop @stack;
	      push @stack, ($b ** $a);
	      last CMD; }
	  if ($cmd eq "sin") { 
	      my $a = pop @stack;
	      push @stack, sin($a/57.29578);
	      last CMD; }
	  if ($cmd eq "cos") { 
	      my $a = pop @stack;
	      push @stack, cos($a/57.29578);
	      last CMD; }
	  else {
	      die "Unknown PS command: $cmd";
	  }
      }
    }

    return @stack;
}
