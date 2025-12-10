(*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: test_functionparser.m                                             *
 *                                                                           *
 *   Create file with results of evaluations of algebraic expersions         *
 *   and make C test file.                                                   *
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

(* === List of tests ========================================================*)

(* Input the function of testing in the following format : 
      {"function_string",{minimal_function_value,maximal_function_value,\
number_of_function_value}       *)
Testsample = {

	(* --- arithmetic operator --- *)

        {"3.45+x", {-2, 2, 5}},
        {"x-4.673", {-200, 200000, 5}},
        {"3*x", {-2, 2, 5}},
        {"x/5.8", {-2, 2, 5}},
        {"x^12-3.345+5", {-2, 2, 5}},
        {"3*x", {-2, 2, 5}},
        {"2.894736*10^2", {-2, 2, 5}},
        {"2.784e-2", {-2,2,5}},
        {"2.784*e",{-2,2,5}},
        {"-4.7285e-7*x",{-2,2,5}},

	(* --- brackets --- *)

	{"3*(x^5-x^4/(1.5-x))", {-2, 2, 5}},

	(* --- relational operators --- *)

	(*
        (* Does not work for Mathematica 5.1 any more! *)
        {"3*(x>1)" ,{-2,2,5}},
        {"3*(x<1)" ,{-2,2,5}},
        {"3*(x>=1)",{-2,2,5}}, 
        {"3*(x<=1)",{-2,2,5}},
        {"3*(x==1)",{-2,2,5}},
        {"3*(2==x)",{-2,2,5}},
        {"3*(x<>1)",{-2,2,5}},
        {"3*(2<>x)",{-2,2,5}},
        {"3*(2!=x)",{-2,2,5}},   
        {"3*(2=x)",{-2,2,5}},
	*)

	(* --- system functions --- *)

	{"exp[-4*X]", {-2, 2, 5}},
	{"log[x]", {1,6,6}},
	{"sqrt[x]", {1,6,6}},
	{"sin[x]", {1,6,6}},
	{"mod[x,3]", {1,6,6}},
	{"sgn[x]", {-2,2,5}},
	{"sec[x]", {-2,2,5}},
	{"exp[-x^2]+log[2]-Pi*sin[x+x*2]", {-5*10^1, 2.3454*10^2,7}},
	{"Sin[x]*3*log[x]", {2, 4, 2}},
	{"abs[x]-3*x", {-2, 2, 5}}

	(*
	(* does not work with Mathematica 3.0 *)	
	{"(sin[ ln[3*x*(cos[ 3*x^3-4.6789/(x+4)])]])-1", {-38.828,454.4*10^3,7}},
	*)

	(*
        (* Does not work for Mathematica 5.1 any more! *)
        {"exp[x^2]*(sin[x*cos[x^2-1]]+1)*((x-3*pi*x)<1)", {-3,7,5}}
	*)
};
 


(* === Set Constants ========================================================*)

(* name of datafile file for running tests *)
DataFile = "t_functionparser.data"; 


(* === Define cosntants and functions for function parser ===================*)

(* --- Constants ----------------------------------------------------------- *)

pi = Pi;
e  = E;

(* --- Functions ----------------------------------------------------------- *)

mod[x_,y_]  := Mod[x,y];
exp[x_]     := Exp[x];
log[x_]     := Log[x];
sin[x_]     := Sin[x];
cos[x_]     := Cos[x];
tan[x_]     := Tan[x];
sec[x_]     := 1/Cos[x];
sqrt[x_]    := Sqrt[x];
abs[x_]     := Abs[x];
sgn[x_]     := Sign[x];    

(* === Define derivatives for these functions (according to function parser) *)

Unprotect[Derivative];

(* --- Relation Operators -------------------------------------------------- *)

Derivative[1][Sign][x_] := 0;

Derivative[1, 0][Unequal][x_, y_] := 0;
Derivative[0, 1][Unequal][x_, y_] := 0;

Derivative[1, 0][Greater][x_, y_] := 0;
Derivative[0, 1][Greater][x_, y_] := 0;

Derivative[1, 0][GreaterEqual][x_, y_] := 0;
Derivative[0, 1][GreaterEqual][x_, y_] := 0;

Derivative[1, 0][Less][x_, y_] := 0;
Derivative[0, 1][Less][x_, y_] := 0;

Derivative[1, 0][LessEqual][x_, y_] := 0;
Derivative[0, 1][LessEqual][x_, y_] := 0;

Derivative[1, 0][Mod][x_, y_] := 0;
Derivative[0, 1][Mod][x_, y_] := 0;


(* --- Functions ----------------------------------------------------------- *)

Derivative[1][Abs][x_] := sgn[x];

(* ------------------------------------------------------------------------- *)

Protect[Derivative];


(* === Write results for an expression into data file ====================== *)

UnurWriteData[expression_,points__] := Module [
	(*	expression ... function term                                 *)
	(*      points     ... list {x_min, x_max, number of points}         *)

	(* local variables                                                   *)
	{
	  funct,   (* string use to compute expression with Mathematica      *)
	  x,       (* argument for function                                  *)
	  xmin,    (* minimal value for x                                    *)
	  xmax,    (* maximal value for x                                    *)
	  xstep    (* step width                                             *)
	},

	(* convert to lower case letters                                     *)
	funct = ToLowerCase[expression];

	(* print function string into data file                              *)
	WriteString[DataFile,
		    "function=",
		    UnurTransformExpression[funct],
		    "\n"];

	(* get values for x                                                  *)
	xmin = points[[1]];
	xmax = points[[2]];
	xstep = (xmax - xmin) / (points[[3]]-1);
	xmax += xstep/2;

	(* print function and its derivative at all given points *)
	Do[ UnurWriteLine[x,funct],{x,xmin,xmax,xstep}];

	(* add blank line *)
	WriteString[DataFile,"\n"]; 

]; (* end of UnurWriteData[] *)


(* --- Transform expression into string for function parser ---------------- *)

UnurTransformExpression[expression_] := Module [
	(*	expression ... function term                                 *)

	(* local variables                                                   *)
	{fstr},

	(* Replace square brackets by parenthesis                            *)
	fstr = StringReplace[ expression, {"[" -> "(", "]" -> ")"}];

	(* return result *)
	Return[fstr];

]; (* end of UnurTransformExpression[] *)

(* --- Transform expression into string for Mathematica computation ----------*)


UnurTransformMathExpression[expression_] := Module [
	(*	expression ... function term                                 *)

	(* local variables                                                   *)
	{fstr},

        (*  replace '<>' -> '!='                                             *)
        fstr = StringReplace[expression, {"<>" -> "!="}];
 
        (*  replace 'and' -> '&&', 'or' -> '||'                              *)
        fstr = StringReplace[fstr,{"and" -> "&&","or" -> "||"}]; 

        (*  replace '=' -> '=='                                              *)
        fstr = StringReplace[fstr,{"=" -> "=="}]; 
        fstr = StringReplace[fstr,{"====" -> "==","<==" -> "<=",">=="->">=",
                                   "!=="  -> "!="}];
	
        (*  replace e.g'2.45e-3' -> '1.45*10^2'                              *)
        fstr = StringReplace[fstr,{"0e" -> "0*10^","1e" -> "1*10^",
                                   "2e" -> "2*10^","3e" -> "3*10^",
                                   "3e" -> "3*10^","4e" -> "4*10^",
                                   "5e" -> "5*10^","6e" -> "6*10^",
                                   "7e" -> "7*10^","8e" -> "8*10^",
                                   "8e" -> "8*10^","9e" -> "9*10^"}];

	(* return result *)
	Return[fstr];

]; (* end of UnurTransformMathExpression[] *)



(* --- Compute function and write line into date file       ---------------- *)

UnurWriteLine[xarg_,funct_] := Module [
	(*      xarg  ... argument                                           *)
	(*	funct ... function term                                      *)

	(* local variables                                                   *)
	{
	  xval, (* numerical value of argument x                             *)
	  fx,   (* value of function at x                                    *)
	  dfx,  (* value of derivative of function at x                      *)
	  fstr  (* function term                                             *)
        },

	(* argument *)
	xval = N[xarg];
	WriteString[DataFile, CForm[xval],"\t"];

        fstr = UnurTransformMathExpression[funct]; 
				    
	(* function *)
	fx = N[ToExpression[fstr]] /. x -> xval /. {True -> 1, False -> 0};
	(* if fx is Complex the output is 'inf' *)
        If[Head[fx]===Complex, 
                              WriteString[DataFile, "inf"    ,"\t"],
                              WriteString[DataFile, CForm[fx],"\t"] ];

	(* derivative *)
	dfx = N[ D[ ToExpression[fstr], x ]] /. x -> xval /. {True -> 1, False -> 0};
        (* if dfx is Complex the output is 'inf' *)	
        If[Head[dfx]===Complex,
                               WriteString[DataFile, "inf"    ,"\n"],
                               WriteString[DataFile, CForm[dfx],"\n"] ];

]; (* end of UnurWriteLine[] *)

(* === Main ================================================================ *)

Do [
   fstr   = Testsample[[i]][[1]];
   points = Testsample[[i]][[2]];
   UnurWriteData[fstr, points],
   {i, 1, Length[Testsample]}
];

(* === Exit ================================================================ *)

Quit[]


