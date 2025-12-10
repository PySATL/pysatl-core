/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: testcounter.c                                                     *
 *                                                                           *
 *   Count function evaluations                                              *
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

#include "testunuran.h"
#include <unuran_config.h>
#include <unur_struct.h>
#include <distr/distr_source.h>

/*---------------------------------------------------------------------------*/
/* count number of function evaluations                                      */

/* -- global varibales ------------------------------------------------------*/

/* pointer for storing distribution object */
struct unur_distr *distr_to_use = NULL;

/* counter for evaluations of PDF and similar functions */
static int counter_pdf = 0;
static int counter_logpdf = 0;
static int counter_cdf = 0;

/* pointer to original functions */
static double (*cont_pdf_to_use)(double x, const struct unur_distr *distr);
static double (*cont_logpdf_to_use)(double x, const struct unur_distr *distr);
static double (*cont_cdf_to_use)(double x, const struct unur_distr *distr);

/*---------------------------------------------------------------------------*/
/* wrapper for functions */

static double 
cont_pdf_with_counter( double x, const struct unur_distr *distr ) {
  ++counter_pdf; return cont_pdf_to_use(x,distr); }

static double 
cont_logpdf_with_counter( double x, const struct unur_distr *distr ) {
  ++counter_logpdf; return cont_logpdf_to_use(x,distr); }

static double 
cont_cdf_with_counter( double x, const struct unur_distr *distr ) {
  ++counter_cdf; return cont_cdf_to_use(x,distr); }

/*---------------------------------------------------------------------------*/
/* set and start counter for PDF calls in parameter object */

int
start_counter_fcalls( UNUR_PAR *par )
{
  /* clear object if necessary */
  stop_counter_fcalls();

  /* make a copy (clone) of the distribution object */
  distr_to_use = _unur_distr_clone(par->distr);

  /* set pointer to distribution object in parameter object to cloned one */
  par->distr = distr_to_use;
  par->distr_is_privatecopy = FALSE;

  /* exchange pointer to PDF etc. with counting wrapper */
  switch (distr_to_use->type) {
  case UNUR_DISTR_CONT:
    /* this only works for continuous distributions */

    /* PDF */
    if (distr_to_use->data.cont.pdf) {
      if (distr_to_use->data.cont.pdf == cont_pdf_with_counter) {
	/* we must avoid this situation as it causes infinite recursion */
	return UNUR_FAILURE;
      } 
      cont_pdf_to_use = distr_to_use->data.cont.pdf; 
      distr_to_use->data.cont.pdf = cont_pdf_with_counter;
    }

    /* logPDF */
    if (distr_to_use->data.cont.logpdf) {
      if (distr_to_use->data.cont.logpdf == cont_logpdf_with_counter) {
	/* we must avoid this situation as it causes infinite recursion */
	return UNUR_FAILURE;
      } 
      cont_logpdf_to_use = distr_to_use->data.cont.logpdf; 
      distr_to_use->data.cont.logpdf = cont_logpdf_with_counter;
    }

    /* CDF */
    if (distr_to_use->data.cont.cdf) {
      if (distr_to_use->data.cont.cdf == cont_cdf_with_counter) {
	return UNUR_FAILURE;
      } 
      cont_cdf_to_use = distr_to_use->data.cont.cdf; 
      distr_to_use->data.cont.cdf = cont_cdf_with_counter;
    }
    
    break;

  default:
    return UNUR_FAILURE;
  }

  /* reset counter */
  reset_counter_fcalls();

  return UNUR_SUCCESS;
}

/*---------------------------------------------------------------------------*/
/* stop counter for PDF calls and clear memory */

int
stop_counter_fcalls(void)
{
  if (distr_to_use) unur_distr_free(distr_to_use);
  distr_to_use = NULL;
  cont_pdf_to_use = NULL;  
  cont_logpdf_to_use = NULL;  
  cont_cdf_to_use = NULL;  
  return UNUR_SUCCESS;
}

/*---------------------------------------------------------------------------*/
/* reset counter to 0 */

void reset_counter_fcalls(void)
{
  counter_pdf = 0;
  counter_logpdf = 0;
  counter_cdf = 0;
}

/*---------------------------------------------------------------------------*/
/* get number of PDF evaluations */

int get_counter_pdf(void) { return counter_pdf; }
int get_counter_logpdf(void) { return counter_logpdf; }
int get_counter_cdf(void) { return counter_cdf; }

/*---------------------------------------------------------------------------*/
