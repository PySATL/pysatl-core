/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  Distribution objects of with multiple of PDFs generated from other       *
 *  distribution objects.                                                    *
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
#include <distr/distr_source.h>
#include "testdistributions.h"

/*---------------------------------------------------------------------------*/
/* Continuous univariate distributions                                       */
/*---------------------------------------------------------------------------*/

static const char distr_name[] = "multPDF";

/*---------------------------------------------------------------------------*/
/* prototypes for wrapper functions */
static double _unur_cdf_mult     (double x, const struct unur_distr *q);
static double _unur_logcdf_mult  (double x, const struct unur_distr *q);
static double _unur_pdf_mult     (double x, const struct unur_distr *q);
static double _unur_logpdf_mult  (double x, const struct unur_distr *q);
static double _unur_dpdf_mult    (double x, const struct unur_distr *q);
static double _unur_dlogpdf_mult (double x, const struct unur_distr *q);

#define DISTR distr->data.cont

#define MULT  params[0]     /* multiplicator */

#define CDF(x)     ((*(q->base->data.cont.cdf))     ((x), q->base))
#define logCDF(x)  ((*(q->base->data.cont.logcdf))  ((x), q->base))
#define PDF(x)     ((*(q->base->data.cont.pdf))     ((x), q->base))
#define logPDF(x)  ((*(q->base->data.cont.logpdf))  ((x), q->base))
#define dPDF(x)    ((*(q->base->data.cont.dpdf))    ((x), q->base))
#define dlogPDF(x) ((*(q->base->data.cont.dlogpdf)) ((x), q->base))

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_multPDF( const struct unur_distr *distr, double mult )
     /*----------------------------------------------------------------------*/
     /* Create an object where the PDF is a multiple of the PDF of 'distr'.  */
     /* `distr' must be a pointer to a univariate continuous distribution.   */
     /*                                                                      */
     /* parameters:                                                          */
     /*   distr ... pointer to  univariate continuous distribution.          */
     /*   mult  ... multiplicator for PDF                                    */
     /*                                                                      */
     /* return:                                                              */
     /*   pointer to distribution object                                     */
     /*                                                                      */
     /* error:                                                               */
     /*   return NULL                                                        */
     /*----------------------------------------------------------------------*/
{
  struct unur_distr *q;

  /* check arguments */
  _unur_check_NULL( distr_name,distr,NULL );

  /* check distribution */
  if (distr->type != UNUR_DISTR_CONT) {
    _unur_error(distr_name,UNUR_ERR_DISTR_INVALID,""); return NULL; }
  COOKIE_CHECK(distr,CK_DISTR_CONT,NULL);

  /* check multiplitor */
  if (mult <= 0.) {
    _unur_error(distr_name,UNUR_ERR_PAR_SET,"mult <= 0"); return NULL; 
  }

  /* get distribution object for generic continuous univariate distribution */
  q = _unur_distr_cont_clone(distr);
  if (!q) return NULL;

  /* this is a derived distribution */
  /* clone base distribution ... */
  q->base = _unur_distr_cont_clone( distr );
  if (!q->base) { _unur_distr_free(q); return NULL; }

  /* pointer to PDF, its derivative, and CDF */
  if (DISTR.cdf)     q->data.cont.cdf     = _unur_cdf_mult;     /* pointer to CDF       */
  if (DISTR.logcdf)  q->data.cont.logcdf  = _unur_logcdf_mult;  /* pointer to CDF       */
  if (DISTR.pdf)     q->data.cont.pdf     = _unur_pdf_mult;     /* pointer to PDF       */
  if (DISTR.dpdf)    q->data.cont.dpdf    = _unur_dpdf_mult;    /* derivative of PDF    */
  if (DISTR.logpdf)  q->data.cont.logpdf  = _unur_logpdf_mult;  /* pointer to logPDF    */
  if (DISTR.dlogpdf) q->data.cont.dlogpdf = _unur_dlogpdf_mult; /* derivative of logPDF */

  /* disable all other pointers */
  q->data.cont.dlogpdf = NULL;          /* pointer to HR                          */

  /* set multiplicator */
  q->data.cont.MULT = mult;
  q->data.cont.n_params = 1; 
 
  /* return pointer to object */
  return q;

} /* end of unur_distr_multPDF() */

/*---------------------------------------------------------------------------*/

double
_unur_cdf_mult( double x, const struct unur_distr *q )
{
  /* check arguments */
  CHECK_NULL( q, UNUR_INFINITY );
  CHECK_NULL( q->base, UNUR_INFINITY );
  CHECK_NULL( q->base->data.cont.pdf, UNUR_INFINITY );

  return CDF(x);

} /* end of _unur_cdf_mult() */

/*---------------------------------------------------------------------------*/

double
_unur_logcdf_mult( double x, const struct unur_distr *q )
{
  /* check arguments */
  CHECK_NULL( q, UNUR_INFINITY );
  CHECK_NULL( q->base, UNUR_INFINITY );
  CHECK_NULL( q->base->data.cont.pdf, UNUR_INFINITY );

  return logCDF(x);

} /* end of _unur_logcdf_mult() */

/*---------------------------------------------------------------------------*/

double
_unur_pdf_mult( double x, const struct unur_distr *q )
{
  /* check arguments */
  CHECK_NULL( q, UNUR_INFINITY );
  CHECK_NULL( q->base, UNUR_INFINITY );
  CHECK_NULL( q->base->data.cont.pdf, UNUR_INFINITY );

  return q->data.cont.MULT * PDF(x);

} /* end of _unur_pdf_mult() */

/*---------------------------------------------------------------------------*/

double
_unur_dpdf_mult( double x, const struct unur_distr *q )
{
  /* check arguments */
  CHECK_NULL( q, UNUR_INFINITY );
  CHECK_NULL( q->base, UNUR_INFINITY );
  CHECK_NULL( q->base->data.cont.dpdf, UNUR_INFINITY );

  return q->data.cont.MULT * dPDF(x);

} /* end of _unur_dpdf_mult() */

/*---------------------------------------------------------------------------*/

double
_unur_logpdf_mult( double x, const struct unur_distr *q )
{
  /* check arguments */
  CHECK_NULL( q, UNUR_INFINITY );
  CHECK_NULL( q->base, UNUR_INFINITY );
  CHECK_NULL( q->base->data.cont.logpdf, UNUR_INFINITY );

  return log(q->data.cont.MULT) + logPDF(x);

} /* end of _unur_logpdf_mult() */

/*---------------------------------------------------------------------------*/

double
_unur_dlogpdf_mult( double x, const struct unur_distr *q )
{
  /* check arguments */
  CHECK_NULL( q, UNUR_INFINITY );
  CHECK_NULL( q->base, UNUR_INFINITY );
  CHECK_NULL( q->base->data.cont.dlogpdf, UNUR_INFINITY );

  return dlogPDF(x);

} /* end of _unur_dlogpdf_mult() */

/*---------------------------------------------------------------------------*/

