/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  Distribution objects of standard distributions without logPDF            *
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

#include <unur_source.h>
#include <unuran.h>
#include "testdistributions.h"

/*---------------------------------------------------------------------------*/
/* Continuous multivariate distributions without logPDF                      */
/*---------------------------------------------------------------------------*/
#define DISTR distr->data.cvec
/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_multinormal_wo_logpdf( int dim, const double *mean, const double *covar )
{
  register struct unur_distr *distr;

  /* get distribution object for beta distribution */
  distr = unur_distr_multinormal_w_marginals( dim, mean, covar );

  /* name of distribution */
  distr->name = "multinormal_wo_logpdf";

  /* disable logPDF */
  DISTR.logpdf   = NULL;
  DISTR.dlogpdf  = NULL;
  DISTR.pdlogpdf = NULL;

  /* return pointer to object */
  return distr;

} /* end of unur_distr_multinormal_wo_logpdf() */

/*---------------------------------------------------------------------------*/
#undef DISTR
/*---------------------------------------------------------------------------*/
