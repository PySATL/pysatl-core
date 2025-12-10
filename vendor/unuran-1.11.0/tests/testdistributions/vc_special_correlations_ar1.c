/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  Distribution objects of multivariate distributions having the            *
 *  correlation matrix of AR(1) process.                                     *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *  Use correlation matrix of AR(1):                                         *
 *                                                                           *
 *      +-                                    -+                             *
 *      |  1       r       r^2     ... r^(d-1) |                             *
 *      |  r       1       r       ... r^(d-2) |                             *
 *      |  r^2     r       1       ... r^(d-3) |                             *
 *      |  ...                     ...         |                             *
 *      |  r^(d-1) r^(d-2) r^(d-3) ... 1       |                             *
 *      +-                                    -+                             *
 *                                                                           *
 *     in this case the inverse matrix is given as                           *
 *                                                                           *
 *      +-                -+                                                 *
 *      |  a b 0 0 ... 0 0 |                                                 *
 *      |  b c b 0 ... 0 0 |                                                 *
 *      |  0 b c b ... 0 0 |                                                 *
 *      |          ...     |                                                 *
 *      |  0 0 0 0 ... c b |                                                 *
 *      |  0 0 0 0 ... b a |                                                 *
 *      +-                -+                                                 *
 *                                                                           *
 *     with                                                                  *
 *                                                                           *
 *       a = 1 / (1-r^2)                                                     *
 *       b = - r / (1-r^2)                                                   *
 *       c = (1+r^2) / (1-r^2)                                               *
 *                                                                           *
 *     and the determinant of the covariance matrix is                       *
 *                                                                           *
 *       det = (1-r^2)^(d-1)                                                 *
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
#include <utils/matrix_source.h> 
#include "testdistributions.h"

/*---------------------------------------------------------------------------*/

#define LOGNORMCONSTANT (distr->data.cvec.norm_constant)
#define NORMCONSTANT    (distr->data.cvec.norm_constant)

/*---------------------------------------------------------------------------*/
/* Set correlation matrix and its inverse of AR(1) process */
static int _unur_vc_set_corrmatrix_ar1( UNUR_DISTR *distr, int dim, double rho ); 

/*---------------------------------------------------------------------------*/

int 
_unur_vc_set_corrmatrix_ar1( UNUR_DISTR *distr, int dim, double rho )
     /*----------------------------------------------------------------------*/
     /* Create and set correlation matrix and its inverse of AR(1) process.  */
     /*                                                                      */
     /* parameters:                                                          */
     /*   distr ... pointer to distribution object                           */
     /*   dim   ... dimension of random vector                               */
     /*   rho   ... correlation between consecutive elements in process      */
     /*                                                                      */
     /* return:                                                              */
     /*   UNUR_SUCCESS ... success                                           */
     /*   UNUR_FAILURE ... failure                                           */
     /*----------------------------------------------------------------------*/
{
#define idx(a,b) ((a)*dim+(b))
  
  double *covar, *covar_inv, *rhop;
  double a, b, c, denominator;
  int i,j;
  int error = UNUR_FAILURE;
  
  /* checking parameters */ 
  if (_unur_iszero(rho)) /* nothing to do */
    return UNUR_SUCCESS;

  /* entries of matrix */
  denominator = 1.-rho*rho;    
  a = 1./denominator;
  b = -rho/denominator;
  c = (1.+rho*rho)/denominator;

  if ( rho<0. || rho>=1. || dim<1)
    /* invalid arguments */
    return UNUR_FAILURE;

  /* allocate memory */
  covar = malloc( dim * dim * sizeof(double) );
  covar_inv = malloc( dim * dim * sizeof(double) );
  rhop = malloc( dim * sizeof(double) );
  if (covar == NULL || covar_inv == NULL || rhop == NULL) {
    if (covar) free (covar);
    if (covar_inv) free (covar_inv);
    if (rhop) free (rhop);
    return UNUR_FAILURE;
  }

  /* compute powers of rho */
  rhop[0] = 1.;
  for (i=1; i<dim; i++) 
    rhop[i] = rhop[i-1] * rho;

  /* create correlation matrix */
  for (i=0; i<dim; i++) {
    for (j=0; j<i; j++)
      covar[idx(i,j)] = rhop[i-j];
    for (j=i; j<dim; j++)
      covar[idx(i,j)] = rhop[j-i];
  }

  /* create inverse of correlation matrix */
  for (i=0; i<dim; i++)
    for (j=0; j<dim; j++) {
      covar_inv[idx(i,j)] = 0.;
      if (i==j) covar_inv[idx(i,j)] = (i==0 || i==(dim-1)) ? a : c;
      if (abs(i-j)==1) covar_inv[idx(i,j)] = b ;
    }
  
  /* set covariance matrix and its inverse */
  if ( (unur_distr_cvec_set_covar( distr, covar ) == UNUR_SUCCESS) &&
       (unur_distr_cvec_set_covar_inv( distr, covar_inv ) == UNUR_SUCCESS) )
    error = UNUR_SUCCESS;

  /* free working space */
  free (covar); free (covar_inv); free (rhop);

  /* return errorcode */
  return error;    

#undef idx
} /* end of _unur_vc_set_corrmatrix_ar1() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_multinormal_ar1(int dim, const double *mean, double rho)
     /*---------------------------------------------------------------------------*/
     /*  Multinormal distribution (corr-matrix from AR(1) process)                */
     /*---------------------------------------------------------------------------*/
{
  struct unur_distr *distr;
  double det_covar;

  /* checking parameters */ 
  if ( rho<0. || rho>=1. || dim<1)
    return NULL;

  /* get distribution object for multinormal distribution */
  distr = unur_distr_multinormal_w_marginals( dim, mean, NULL );

  /* set the name of distribution */
  unur_distr_set_name(distr, "multinormal_ar1");
  
  /* set the correlation matrix and its inverse */
  if (_unur_vc_set_corrmatrix_ar1(distr, dim, rho) != UNUR_SUCCESS) {
    /* error */
    unur_distr_free(distr); distr = NULL; return NULL;
  }
  
  /* compute normalization constant */
  det_covar = _unur_matrix_determinant(distr->dim, distr->data.cvec.covar);
  LOGNORMCONSTANT = - ( distr->dim * log(2 * M_PI) + log(det_covar) ) / 2.;

  /* return pointer to object */
  return distr;

} /* end of unur_distr_multinormal_ar1() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_multicauchy_ar1(int dim, const double *mean, double rho)
     /*---------------------------------------------------------------------------*/
     /*  Multinormal distribution (corr-matrix from AR(1) process)                */
     /*---------------------------------------------------------------------------*/
{
  struct unur_distr *distr;
  double det_covar;
  
  /* checking parameters */ 
  if ( rho<0. || rho>=1. || dim<1)
    return NULL;

  /* get distribution object for multicauchy distribution */
  distr = unur_distr_multicauchy_w_marginals( dim, mean, NULL );

  /* set the name of distribution */
  unur_distr_set_name(distr, "multicauchy_ar1");
  
  /* set the correlation matrix and its inverse */
  if (_unur_vc_set_corrmatrix_ar1(distr, dim, rho) != UNUR_SUCCESS) {
    /* error */
    unur_distr_free(distr); distr = NULL; return NULL;
  }
  
  /* compute normalization constant */
  det_covar = _unur_matrix_determinant(distr->dim, distr->data.cvec.covar);
  LOGNORMCONSTANT = _unur_SF_ln_gamma((distr->dim+1)/2.) 
                  - ( (distr->dim+1) * log(M_PI) + log(det_covar) ) / 2.;

  /* return pointer to object */
  return distr;

} /* end of unur_distr_multicauchy_ar1() */

/*---------------------------------------------------------------------------*/

struct unur_distr *
unur_distr_multistudent_ar1(int dim, double df, const double *mean, double rho)
     /*---------------------------------------------------------------------------*/
     /*  Multistudent distribution (corr-matrix from AR(1) process)               */
     /*---------------------------------------------------------------------------*/
{
  struct unur_distr *distr;
  double det_covar;
  
  /* checking parameters */ 
  if ( rho<0. || rho>=1. || dim<1)
    return NULL;

  /* get distribution object for multistudent distribution */
  distr = unur_distr_multistudent_w_marginals( dim, df, mean, NULL );

  /* set the name of distribution */
  unur_distr_set_name(distr, "multistudent_ar1");
  
  /* set the correlation matrix and its inverse */
  if (_unur_vc_set_corrmatrix_ar1(distr, dim, rho) != UNUR_SUCCESS) {
    /* error */
    unur_distr_free(distr); distr = NULL; return NULL;
  }
  
  /* compute normalization constant */
  det_covar = _unur_matrix_determinant(distr->dim, distr->data.cvec.covar);
  LOGNORMCONSTANT = _unur_SF_ln_gamma((distr->dim+df)/2.) - _unur_SF_ln_gamma(df/2.)
                  - ( distr->dim * log(df*M_PI) + log(det_covar) ) / 2.;

  /* return pointer to object */
  return distr;

} /* end of unur_distr_multistudent_ar1() */

/*---------------------------------------------------------------------------*/
