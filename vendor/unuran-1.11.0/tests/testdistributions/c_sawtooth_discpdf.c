/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  Sawtooth distribution with discontinuous PDF                             *
 *                                                                           *
 *  pdf:               / 1                 if x is integer != 0              *
 *             f(x) = <                                                      *
 *                     \ |x| - floor(|x|)  otherwise                         *
 *  domain:    closed interval                                               *
 *  constant:  depends on interval                                           *
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
#include <distr/cont.h>
#include <distributions/unur_stddistr.h>
#include <utils/unur_fp_source.h>
#include "testdistributions.h"

/*---------------------------------------------------------------------------*/
static const char distr_name[] = "sawtooth_discpdf";

/* parameters */
#define DISTR distr->data.cont
#define bd_left  (params[0])
#define bd_right (params[1])

/* function prototypes                                                       */
static double _unur_pdf_sawtooth_discpdf( double x, const UNUR_DISTR *distr );
static double _unur_dpdf_sawtooth_discpdf( double x, const UNUR_DISTR *distr );
static double _unur_cdf_sawtooth_discpdf( double x, const UNUR_DISTR *distr );

static int _unur_set_params_sawtooth_discpdf( UNUR_DISTR *distr, const double *params, int n_params );
static int _unur_upd_area_sawtooth_discpdf( UNUR_DISTR *distr );

/*---------------------------------------------------------------------------*/

double
_unur_pdf_sawtooth_discpdf( double x, const UNUR_DISTR *distr ATTRIBUTE__UNUSED )
{
  double pdf;

  if (_unur_iszero(x)) 
    return 0.;

  /* else */
  x = fabs(x);
  pdf = x - floor(x);
  return _unur_iszero(pdf) ? 1. : pdf;
} /* end of _unur_pdf_sawtooth_discpdf() */

/*---------------------------------------------------------------------------*/
  
double
_unur_dpdf_sawtooth_discpdf( double x, const UNUR_DISTR *distr ATTRIBUTE__UNUSED )
{
  return (x<0.) ? -1. : 1.;
} /* end of _unur_dpdf_sawtooth_discpdf() */

/*---------------------------------------------------------------------------*/

static double integral_0_x( double x )
{
  double absx = fabs(x);
  double intx = floor(absx);
  return 0.5*(intx + (absx-intx)*(absx-intx));
} /* end of integral_0_x() */

static double integral_x_y( double x, double y )
{
  double ix = integral_0_x(x);
  double iy = integral_0_x(y);
  double sign = 1.;

  if (x>y) { double tmp=x; x=y; y=tmp; sign=-1.; }

  if (x<0. && y<0.)  return sign*(ix - iy);
  if (x<0. && y>=0.) return sign*(ix + iy);
  /* otherwise */
  return sign*(iy -ix);
} /* end of integral_x_y() */

/*---------------------------------------------------------------------------*/

double
_unur_cdf_sawtooth_discpdf( double x, const UNUR_DISTR *distr )
{
  return (integral_x_y(DISTR.domain[0],x) / integral_x_y(DISTR.domain[0],DISTR.domain[1]));
} /* end of _unur_cdf_sawtooth_discpdf() */

/*---------------------------------------------------------------------------*/

int
_unur_upd_area_sawtooth_discpdf( UNUR_DISTR *distr )
{
  double area = integral_x_y(DISTR.domain[0],DISTR.domain[1]);
  if (!_unur_isfinite(area)) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"invalid domain");
    return UNUR_ERR_DISTR_DOMAIN;
  }
 
  DISTR.area = area;
  return UNUR_SUCCESS;
} /* end of _unur_upd_area_sawtooth_discpdf() */

/*---------------------------------------------------------------------------*/

int
_unur_set_params_sawtooth_discpdf( UNUR_DISTR *distr, const double *params, int n_params )
{
  /* check number of parameters for distribution */
  if (n_params != 2) {
    _unur_warning(distr_name,UNUR_ERR_DISTR_NPARAMS,"");
    return UNUR_ERR_DISTR_NPARAMS;
  }
  CHECK_NULL(params,UNUR_ERR_NULL);

  /* check parameters */
  if (! (_unur_FP_less(bd_left,bd_right) || _unur_isfinite(bd_left) || _unur_isfinite(bd_right)) ) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"invalid domain");
    return UNUR_ERR_DISTR_DOMAIN;
  }

  /* set domain */
  DISTR.domain[0] = bd_left;
  DISTR.domain[1] = bd_right;

  /* update area below PDF */
  if (_unur_upd_area_sawtooth_discpdf(distr) != UNUR_SUCCESS) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"invalid domain");
    return UNUR_ERR_DISTR_DOMAIN;
  }

  return UNUR_SUCCESS;
} /* end of _unur_set_params_sawtooth_discpdf() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_sawtooth_discpdf( const double *params, int n_params )
     /* boundary of domain as parameters */
{
  register struct unur_distr *distr;

  if (! (_unur_FP_less(bd_left,bd_right) || _unur_isfinite(bd_left) || _unur_isfinite(bd_right)) ) {
    _unur_error(distr_name,UNUR_ERR_DISTR_DOMAIN,"invalid domain");
    return NULL;
  }

  /* get new (empty) distribution object */
  distr = unur_distr_cont_new();

  /* set distribution id */
  distr->id = UNUR_DISTR_GENERIC;

  /* name of distribution */
  distr->name = distr_name;
                
  /* how to get special generators */
  /* DISTR.init = NULL; */

  /* functions */
  DISTR.pdf     = _unur_pdf_sawtooth_discpdf;     /* pointer to PDF                  */
  DISTR.dpdf    = _unur_dpdf_sawtooth_discpdf;    /* pointer to derivative of PDF    */
  DISTR.cdf     = _unur_cdf_sawtooth_discpdf;     /* pointer to CDF                  */

  /* indicate which parameters are set */
  distr->set = ( UNUR_DISTR_SET_DOMAIN |
		 UNUR_DISTR_SET_STDDOMAIN |
		 UNUR_DISTR_SET_PDFAREA );
                
  /* set parameters for distribution */
  if (_unur_set_params_sawtooth_discpdf(distr,params,n_params)!=UNUR_SUCCESS) {
    free(distr); return NULL;
  }

  /* log of normalization constant */
  /* NORMCONSTANT = 1.; */

  /* function for setting parameters and updating domain */
  DISTR.set_params = _unur_set_params_sawtooth_discpdf;

  /* function for updating derived parameters */
  DISTR.upd_area  = _unur_upd_area_sawtooth_discpdf; /* funct for computing area */

  /* return pointer to object */
  return distr;

} /* end of unur_distr_sawtooth_discpdf() */

/*---------------------------------------------------------------------------*/
#undef DISTR
#undef bd_left
#undef bd_right
/*---------------------------------------------------------------------------*/
