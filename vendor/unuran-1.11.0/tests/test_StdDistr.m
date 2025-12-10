(*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: test_StdDistr.m                                                   *
 *                                                                           *
 *   Create file with results of CDF, PDF and derivatives of PDF             *
 *   for various distributions and make C test file.                         *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2006 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *                                                                           *
 *   This program is free software; you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation; either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program; if not, write to the                           *
 *   Free Software Foundation, Inc.,                                         *
 *   59 Temple Place, Suite 330, Boston, MA 02111-1307, USA                  *
 *                                                                           *
 *****************************************************************************)

(* === Set Constants ========================================================*)

If [ Environment["srcdir"] != $Failed, 
	SrcDir = Environment["srcdir"],
(* Else *)
        SrcDir =  "./" ];

(* name of C file for running tests *)
SrcFileName = "t_StdDistr.c";

(* name of data file for running tests *)
DataFileName = "t_StdDistr.data";

(* sample size for tests *)
RunSampleSize = 10000;

(* constants *)
isCONT = 1;
isDISCR = 2;

(* === Declare additional distributions ==================================== *)

(* --- Defaults (to avoid error with C program) ---------------------------- *)

(* --- Extreme Value II Distribution --------------------------------------- *)

ExtremeValueIIDistribution/: Domain[ExtremeValueIIDistribution[k_,zeta_,theta_]] := 
	Interval[{zeta, Infinity}];

ExtremeValueIIDistribution/: PDF[ExtremeValueIIDistribution[k_,zeta_,theta_], x_] :=
	If [ x > zeta,
		Exp[ -((x-zeta)/theta)^(-k)] * ((x-zeta)/theta)^(-k-1) * k/theta,
	(* else *)
		0
	];

ExtremeValueIIDistribution/: CDF[ExtremeValueIIDistribution[k_,zeta_,theta_], x_] :=
	If [ x > zeta,
		Exp[ -((x-zeta)/theta)^(-k)],
	(* else *)
		0
	];

ExtremeValueIIDistribution/: Random[ExtremeValueIIDistribution[k_,zeta_,theta_]] :=
	zeta + theta * Exp[ -Log[ -Log[Random[]] ]/k ];

(* --- Lomax Distribution -------------------------------------------------- *)

LomaxDistribution/: Domain[LomaxDistribution[___]] := 
	Interval[{0,1}];

LomaxDistribution/: PDF[LomaxDistribution[a_,C_], x_] :=
	(x+C)^(-(a+1)) * a * C^a;

LomaxDistribution/: CDF[LomaxDistribution[a_,C_], x_] :=
	1 - C^a/(C + x)^a;

LomaxDistribution/: Random[LomaxDistribution[a_,C_]] :=
	-C + (-(C^a/(-1 + Random[])))^(1/a);

(* --- Powerexponential Distribution --------------------------------------- *)

PowerexponentialDistribution/: Domain[PowerexponentialDistribution[___]] := 
	Interval[{-Infinity, Infinity}];

PowerexponentialDistribution/: PDF[PowerexponentialDistribution[r_], x_] :=
	If [ x>0, 
		Exp[-x^r]/(2 Gamma[1+1/r]),
	(* Else *)
		Exp[-(-x)^r]/(2 Gamma[1+1/r]) ];

PowerexponentialDistribution/: CDF[PowerexponentialDistribution[r_], x_] := 
	If [ x<0,
		Gamma[1/r, (-x)^r] / (2*r*Gamma[1 + 1/r]),
	(* Else *)
		1/2 + (Gamma[1 + 1/r] - Gamma[1/r, x^r]/r) / (2 Gamma[1+1/r]) ];

PowerexponentialDistribution/: Random[PowerexponentialDistribution[r_]] := 
	(* Warning!! This not a generator for the Powerexponential Distribution!!       *)
	(* FindRoot[ PDF[PowerexponentialDistribution[r],x] == Random[],{x,0.3}][[1,2]] *)
	(*  is correct but very slow. So we use a Laplace random variate instead !!     *)
	If[ Random[] < 0.5, 1, -1] * If[ Random[] < Max[0.3,(1-1/r)], Random[], 1-Log[Random[]]/r ];

(* --- Triangular Distribution --------------------------------------------- *)

(* We use our own version of the triangular distribution.                    *)
Unprotect[TriangularDistribution];
Clear[TriangularDistribution];

TriangularDistribution/: Domain[TriangularDistribution[___]] := 
	Interval[{0,1}];

TriangularDistribution/: PDF[TriangularDistribution[H_], x_] :=
	Which [ x > 0 && x <= H,
			2*x / H,
		x > H && x < 1,
			2*(1-x) / (1-H),
		True,
			0
	];

TriangularDistribution/: CDF[TriangularDistribution[H_], x_] := 
	Which [ x <= 0,
			0,
		x > 0 && x <= H,
			x*x / H,
		x > H && x < 1,
			(H + x * (x-2))/(H-1),
		True,
			1
	];

TriangularDistribution/: Random[TriangularDistribution[H_]] := 
	(* Warning!! This not a generator for the Triangular Distribution!!       *)
	(* We simply use uniform distribution!!                                   *)
	Random[];

(* === End of Distributions ================================================ *)

(* === Format and write output ============================================= *)

(* --- Write result for parameter/argument combination into file ----------- *)

UnurTestDistrResultLine[stream_, distr_, dtype_, fparams__, x_] := Module [
	(*	stream  ... output stream                                    *)
	(*   	distr   ... distribution                                     *)
	(*	dtype   ... type of distribution (CONT|DISCR|...)            *)
	(*	fparams ... parameters for distribution                      *)
	(*	x       ... argument where 'distr' should be evaluated       *)

	(* local variables *)
	{i,Fx},

	(* number of parameters *)
	WriteString[stream, Length[fparams]," "];

	(* list of parameters *)
	Table[
		WriteString[stream, CForm[ N[ fparams[[i]] ]]," "],
		{i,1,Length[fparams]} ];

	(* argument *)
	WriteString[stream, CForm[x]," "];

	(* CDF *)
	WriteString[stream, CForm[ N[ CDF[ Apply[distr,fparams], x ]]]," "];
	
	(* PDF *)
	WriteString[stream, CForm[ N[ PDF[ Apply[distr,fparams], x ]]]," "];

	(* derivative of PDF *)
	Switch [ dtype,
		isCONT,
		        Fx = N[ 0. + D[ PDF[ Apply[distr,fparams], t ], t] /. t->x],
		isDISCR, (* discrete distribution: no derivative *)
			Fx = 0,
		_, (* default: should not happen *)
			Fx = 0
	];
	WriteString[stream, CForm[ Fx ]];

	(* end *)
	WriteString[stream, "\n"];

]; (* end of UnurTestDistrResultLine[] *)

(* --- make file with results for given distribution ----------------------- *)

UnurTestDistrResultFile[dname_, dtype_, datafile_, fparbd__, size_, distribution___] := Module [ 
	(*	dname   ... UNU.RAN name of distribution                     *)
	(*	dtype   ... type of distribution (CONT|DISCR|...)            *)
	(*	srcfile ... output stream for C file running tests           *)
	(*	datafile... output stream for data for running tests         *)
	(*	fparbd  ... bounds for parameters for distribution           *)
	(*		    (use isDISCR as 3rd entry in list for            *)
	(*		    discrete parameters)                             *)
	(*	size    ... size of sample                                   *)
	(*	distribution ... (optional)				     *)
	
	(* local variables *)
	{distr, distrAPI, distrstring, i, j, nfparams, fparams, x},

	(* name of distribution for Mathematica *)
	If [ Length[{distribution}] > 0,
		(* distribution given as third argument *)
		distr = distribution,
	(* else *)
		(* get distribution from dname *)
		distrstring = ToUpperCase[ StringTake[dname,1] ]
				<> StringDrop[dname,1] 
				<> "Distribution";
		distr = ToExpression[distrstring];
	];

	(* number of parameters for distribution *)
	nfparams = Length[fparbd];

	(* compose string for UNU.RAN string API *)
	distrAPI = dname <> "(";
	For [ i=0, i<nfparams, i++,
		distrAPI = distrAPI <> ToString[ N[fparbd[[i+1,2]]] ];
		If [ i<nfparams-1, distrAPI = distrAPI <> ", " ];
	];
	distrAPI = distrAPI <> ")";
	Print[ distrAPI ];

	(* print distribution into data file *)
	WriteString[datafile,"distr=" <> distrAPI <> "\n"];

	(* print results into data file *)
	Do[
		(* make parameters for distribution at random *)
		fparams = 
			Table[ 
				If [ Length[fparbd[[j]]]>2 && fparbd[[j,3]] == isDISCR,
					Round[ fparbd[[j,1]] + Random[] * (fparbd[[j,2]]-fparbd[[j,1]]) ],
				(* else: isCONT *)
					N[ fparbd[[j,1]] + Random[] * (fparbd[[j,2]]-fparbd[[j,1]]) ]
				],
			{j,nfparams} ];	

		(* make an argument at random (use importance sampling) *)
		x = Random[ Apply[distr,fparams] ];
		If [ dtype == isCONT, x = N[x] ];

		(* print line with this parameters *)
		UnurTestDistrResultLine[datafile,distr,dtype,fparams,x],

	{i,size} ];

	(* empty line as separator between distributions *)
	WriteString[datafile,"\n"];

]; (* end of UnurTestDistrResultFile[] *)


(* === Start of Main ======================================================= *)

(* Open data file *)
datafile = OpenWrite[ DataFileName ];

(* --- List of Continuous Distributions ------------------------------------ *)

(* Beta *)
fparams = {{1,10}, {1,100}};
UnurTestDistrResultFile["beta", isCONT, datafile, fparams, RunSampleSize];

(* Cauchy *)
fparams = {{-100,100},{1/100,100}};
UnurTestDistrResultFile["cauchy", isCONT, datafile, fparams, RunSampleSize];

(* Chi *)
fparams = {{1,100}};
UnurTestDistrResultFile["chi", isCONT, datafile, fparams, RunSampleSize];

(* Chisquare *)
fparams = {{1,100}};
UnurTestDistrResultFile["chisquare", isCONT, datafile, fparams, RunSampleSize, ChiSquareDistribution];

(* Exponential -- parameters differ *)
fparams = {{1/100,100}};
ed[mu_] = ExponentialDistribution[1/mu];
UnurTestDistrResultFile["exponential", isCONT, datafile, fparams, RunSampleSize, ed];

(* ExtremeValue I *)
fparams = {{-100,100},{1/100,100}};
UnurTestDistrResultFile["extremeI",isCONT, datafile, fparams, RunSampleSize, ExtremeValueDistribution];

(* ExtremeValue II *)
fparams = {{1/100,100}, {-100,100}, {1/100,100}};
UnurTestDistrResultFile["extremeII",isCONT, datafile, fparams, RunSampleSize, ExtremeValueIIDistribution];

(* F *)
fparams = {{1/100,10}, {1/100,10}};
UnurTestDistrResultFile["F", isCONT, datafile, fparams, RunSampleSize, FRatioDistribution];

(* Gamma *)
fparams = {{1/2,10}, {1/100,100}};
UnurTestDistrResultFile["gamma", isCONT, datafile, fparams, RunSampleSize];

(* InverseGaussianDistribution *)
(* Remark: the CDF values differ for mu < 0 *)
fparams = {{1,100}, {1/100,10}};
UnurTestDistrResultFile["ig", isCONT, datafile, fparams, RunSampleSize, InverseGaussianDistribution];

(* Laplace *)
(* Remark: dPDF is computed incorrectly! *)
fparams = {{-100,100}, {1/100,100}};
UnurTestDistrResultFile["laplace", isCONT, datafile, fparams, RunSampleSize];

(* Lomax *)
fparams = {{1/100,100}, {1/100,100}};
UnurTestDistrResultFile["lomax", isCONT, datafile, fparams, RunSampleSize];

(* Logistic *)
fparams = {{-100,100},{1/100,100}};
UnurTestDistrResultFile["logistic", isCONT, datafile, fparams, RunSampleSize];

(* Normal *)
fparams = {{-100,100}, {1/100,100}};
UnurTestDistrResultFile["normal", isCONT, datafile, fparams, RunSampleSize];

(* Pareto *)
fparams = {{1/100,100},{1/100,100}};
UnurTestDistrResultFile["pareto", isCONT, datafile, fparams, RunSampleSize];

(* Powerexponential *)
fparams = {{1/1000,3}};
UnurTestDistrResultFile["powerexponential", isCONT, datafile, fparams, RunSampleSize];

(* Rayleigh *)
fparams = {{1/100,100}};
UnurTestDistrResultFile["rayleigh", isCONT, datafile, fparams, RunSampleSize];

(* Student *)
fparams = {{1/100,100}};
UnurTestDistrResultFile["student", isCONT, datafile, fparams, RunSampleSize, StudentTDistribution];

(* Triangular *)
fparams = {{0,1}};
UnurTestDistrResultFile["triangular", isCONT, datafile, fparams, RunSampleSize];

(*
(* Uniform *)
(* Disabled! since Mathematica 3.0 computes wrong values for the derivative *)
fparams = {{-100,1}, {1001/1000,100}};
UnurTestDistrResultFile["uniform", isCONT, datafile, fparams, RunSampleSize];
*)

(* Weibull *)
fparams = {{1/2,10},{1/100,100}};
UnurTestDistrResultFile["weibull", isCONT, datafile, fparams, RunSampleSize];

(* --- List of Discrete Distributions -------------------------------------- *)

(* Binomial *)
fparams = {{2,1000,isDISCR}, {1/1000,999/1000}};
UnurTestDistrResultFile["binomial", is_DISCR, datafile, fparams, RunSampleSize];

(* Geometric *)
fparams = {{1/1000,999/1000}};
UnurTestDistrResultFile["geometric", is_DISCR, datafile, fparams, RunSampleSize];

(* Hypergeometric; order of parameters is different *)
hgd[N_,M_,n_] = HypergeometricDistribution[n,M,N];
fparams = {{100,1000,isDISCR}, {1,99,isDISCR}, {1,99,isDISCR}};
UnurTestDistrResultFile["hypergeometric", is_DISCR, datafile, fparams, RunSampleSize, hgd];

(* Logarithmic *)
fparams = {{1/1000,999/1000}};
UnurTestDistrResultFile["logarithmic", is_DISCR, datafile, fparams, RunSampleSize, LogSeriesDistribution];

(* NegativeBinomial; order of parameters is different *)
nb[p_,n_] = NegativeBinomialDistribution[n,p];
fparams = {{1/1000,999/1000}, {1,100,isDISCR}};
UnurTestDistrResultFile["negativebinomial", is_DISCR, datafile, fparams, RunSampleSize, nb];

(* Poisson *)
fparams = {{1/100,100}};
UnurTestDistrResultFile["poisson", is_DISCR, datafile, fparams, RunSampleSize];


(* --- Done ---------------------------------------------------------------- *)

(* close data file *)
Close[datafile];

(* === Exit ================================================================ *)

Quit[]
