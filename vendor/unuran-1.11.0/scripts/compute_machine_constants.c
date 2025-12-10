/*****************************************************************************
 *                                                                           *
 *          unuran -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      compute_machine_constants.c                                  *
 *                                                                           *
 *   Compute some machine constants and print to stdout                      *
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
 *****************************************************************************/

/*---------------------------------------------------------------------------*/

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*---------------------------------------------------------------------------*/

/** TODO:
#define MAXLGM 2.556348e305
**/

/*---------------------------------------------------------------------------*/

int main(void)
{

  printf("/** File automatically created by scripts/compute_machine_constants         **/\n\n");

  printf("/*****************************************************************************\n");
  printf(" *                                                                           *\n");
  printf(" *          unuran -- Universal Non-Uniform Random number generator          *\n");
  printf(" *                                                                           *\n");
  printf(" *****************************************************************************/\n");

  printf("\n");

  printf("/*---------------------------------------------------------------------------*/\n");
  printf("#ifndef UNUR_FP_CONST_SOURCE_H_SEEN\n");
  printf("#define UNUR_FP_CONST_SOURCE_H_SEEN\n");
  printf("/*---------------------------------------------------------------------------*/\n\n");


  /* Epsilon for comparision of two doubles:                                   */
  /* Two doubles are considered equal if there relative difference is          */
  /* less than UNUR_EPSILON (see file `./methods/source_fp.h' for details).    */
  /* Use 100 times DBL_EPSILON.                                                */

  printf("/* maximal relative error when testing equality of two doubles */\n");
  printf("#define UNUR_EPSILON  %.30g\n\n", 100.*DBL_EPSILON);


  /* Square root of machine epsilon. It is used to compare two doubles         */
  /* when round-off errors have to be considered                               */
  /* (see file `./methods/source_fp.h' for details).                           */

  printf("/* square root of DBL_EPSILON */\n");  
  printf("#define UNUR_SQRT_DBL_EPSILON   %.30g\n\n", sqrt(DBL_EPSILON));

  /* Log of machine epsilon.                                                   */
  /* Negative of least number x such that exp(x)-1. == exp(x)                  */

  printf("/* log of DBL_EPSILON */\n");  
  printf("#define UNUR_LOG_DBL_EPSILON   %.30g\n\n", log(DBL_EPSILON));

  /* Constants used by CEPHES functions */

  printf("/* the machine roundoff error */\n");  
  printf("#define MACHEP  %.30g\n\n", DBL_EPSILON/2.);

  printf("/* largest argument for exp() */\n");
  printf("#define MAXLOG  %.30g\n\n", log(DBL_MAX)); 

  printf("/* smallest argument for exp() without underflow */\n");
  printf("#define MINLOG  %.30g\n\n", log(DBL_MIN)); 

  printf("/* the maximal number that pow(x,x-0.5) has no overflow */\n");
  printf("/* we use a (very) conservative portable bound          */\n");
  printf("#define MAXSTIR  %.30g\n\n", log(DBL_MAX) / log(log(DBL_MAX)) );

  printf("/*---------------------------------------------------------------------------*/\n");
  printf("#endif  /* UNUR_FP_CONST_SOURCE_H_SEEN */\n");
  printf("/*---------------------------------------------------------------------------*/\n");

  exit (0);
}

/*---------------------------------------------------------------------------*/
