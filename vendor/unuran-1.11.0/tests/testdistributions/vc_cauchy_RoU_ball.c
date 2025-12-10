/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  Distribution objects of cauchy distribution whch becomes a ball when     *
 *  the ratio-of-uniforms transformation is applied.                         *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  PDF(x) = (2 / (1+||x||^2) )^{dim+1}                                      *
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
#include <distr/distr_source.h>
#include <distr/cvec.h>
#include <distributions/unur_stddistr.h>
#include "testdistributions.h"

/*---------------------------------------------------------------------------*/

static const char distr_name[] = "multicauchy-RoU-ball";

/*---------------------------------------------------------------------------*/
/* parameters */

#define DISTR distr->data.cvec

/*---------------------------------------------------------------------------*/
/* function prototypes                                                       */

static double _unur_pdf_multicauchy_RoU_ball( const double *x, UNUR_DISTR *distr );
static double _unur_logpdf_multicauchy_RoU_ball( const double *x, UNUR_DISTR *distr );
static int _unur_dlogpdf_multicauchy_RoU_ball( double *result, const double *x, UNUR_DISTR *distr );

/*---------------------------------------------------------------------------*/

double
_unur_pdf_multicauchy_RoU_ball( const double *x, UNUR_DISTR *distr )
{ 
  return exp(_unur_logpdf_multicauchy_RoU_ball( x, distr ));
} /* end of _unur_pdf_multicauchy_RoU_ball() */

/*---------------------------------------------------------------------------*/

double
_unur_logpdf_multicauchy_RoU_ball( const double *x, UNUR_DISTR *distr )
{ 
  int i,dim;
  double normsq;

  dim = distr->dim;
  for (i=0, normsq=0.; i<dim; i++) normsq += x[i]*x[i];
  return (dim+1.)*(M_LN2 - log(1.+normsq));
} /* end of _unur_logpdf_multicauchy_RoU_ball() */

/*---------------------------------------------------------------------------*/

int
_unur_dlogpdf_multicauchy_RoU_ball( double *result, const double *x, UNUR_DISTR *distr )
{
  int i,dim;
  double normsq;

  dim = distr->dim;
  for (i=0, normsq=0.; i<dim; i++) 
    normsq += x[i]*x[i];

  for (i=0; i<dim; i++)
    result[i] = -2.*(dim+1.)*x[i] / (1.+normsq);

  return UNUR_SUCCESS; 

} /* end of _unur_dlogpdf_multicauchy_RoU_ball() */

/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**                                                                         **/
/**  Make distribution object                                               **/
/**                                                                         **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_multicauchy_RoU_ball( int dim )
{
  struct unur_distr *distr;
  int i;
  
  /* get new (empty) distribution object */
  distr = unur_distr_cvec_new(dim);

  /* check new parameter for generator */
  if (distr == NULL) {
    /* error: dim < 1 */
    return NULL;
  }

  /* set distribution id */
  distr->id = UNUR_DISTR_MCAUCHY;

  /* name of distribution */
  distr->name = distr_name;

  /* how to get special generators */
  DISTR.init = NULL;

  /* functions */
  DISTR.pdf     = _unur_pdf_multicauchy_RoU_ball;       /* pointer to PDF */
  DISTR.logpdf  = _unur_logpdf_multicauchy_RoU_ball;    /* pointer to logPDF */
  DISTR.dpdf    = _unur_distr_cvec_eval_dpdf_from_dlogpdf;  /* pointer to derivative of PDF */
  DISTR.dlogpdf = _unur_dlogpdf_multicauchy_RoU_ball;   /* pointer to derivative of logPDF */

  /* copy other parameters of distribution */
  /* none */

  /* number of other parameters */
  /* DISTR.n_params = 0;  ... default */

  /* domain */

  /* log of normalization constant */

  /* mode */
  DISTR.mode = _unur_xmalloc( distr->dim * sizeof(double) );
  for (i=0; i<dim; i++)
    DISTR.mode[i] = 0.;

  /* volume below p.d.f. */

  /* indicate which parameters are set (additional to mean and covariance) */
  distr->set |= ( UNUR_DISTR_SET_STDDOMAIN |
		  UNUR_DISTR_SET_MODE );

  /* return pointer to object */
  return distr;

} /* end of unur_distr_multicauchy_RoU_ball() */

/*---------------------------------------------------------------------------*/
#undef DISTR
/*---------------------------------------------------------------------------*/
