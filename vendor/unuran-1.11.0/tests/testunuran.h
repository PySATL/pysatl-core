/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE:      testunuran.h                                                 *
 *                                                                           *
 *   Prototypes for common test routines                                     *
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <config.h>
#include <unuran.h>
#include <unuran_tests.h>
#include "testdistributions/testdistributions.h"

/* macros and functions for handling floats */
#include <utils/unur_fp_source.h>
#include <utils/unur_fp_const_source.h>

/*---------------------------------------------------------------------------*/
/* global variables                                                          */

/* tresholds */
#define PVAL_LIMIT (1.e-3)                  /* treshold for p-value for stat. test */
#define DEFAULT_CHI2_FAILURES_TOLERATED (3) /* tolerated number of failed tests    */

/* chisquare test */
#define CHI_TEST_INTERVALS 100  /* number of intervals for chi^2 test        */
#define CHI_TEST_VERBOSITY 0    /* verbosity level for chi^2 test: 0 | 1 | 2 */

/*---------------------------------------------------------------------------*/
/* constants                                                                 */

#ifndef TRUE
#define TRUE   (1)
#endif

#ifndef FALSE
#define FALSE  (0)
#endif

#ifndef M_LN10
#define M_LN10  2.302585092994046
#endif

/*---------------------------------------------------------------------------*/
/* define macros for GCC attributes                                          */

#ifdef __GNUC__
#  define ATTRIBUTE__UNUSED        __attribute__ ((unused))
#else
#  define ATTRIBUTE__UNUSED
#endif

/*---------------------------------------------------------------------------*/
/* suppress GCC warnings                                                     */
/* from Jonathan Wakely (copied from Patrick Horgan                          */ 
/*   http://dbp-consulting.com/tutorials/SuppressingGCCWarnings.html )       */

#if ((__GNUC__ * 100) + __GNUC_MINOR__) >= 402
#define GCC_DIAG_STR(s) #s
#define GCC_DIAG_JOINSTR(x,y) GCC_DIAG_STR(x ## y)
#  define GCC_DIAG_DO_PRAGMA(x) _Pragma (#x)
#  define GCC_DIAG_PRAGMA(x) GCC_DIAG_DO_PRAGMA(GCC diagnostic x)
#  if ((__GNUC__ * 100) + __GNUC_MINOR__) >= 406
#    define GCC_DIAG_OFF(x) GCC_DIAG_PRAGMA(push) \
        GCC_DIAG_PRAGMA(ignored GCC_DIAG_JOINSTR(-W,x))
#    define GCC_DIAG_ON(x) GCC_DIAG_PRAGMA(pop)
#  else
#    define GCC_DIAG_OFF(x) GCC_DIAG_PRAGMA(ignored GCC_DIAG_JOINSTR(-W,x))
#    define GCC_DIAG_ON(x) GCC_DIAG_PRAGMA(warning GCC_DIAG_JOINSTR(-W,x))
#  endif
#else
#  define GCC_DIAG_OFF(x)
#  define GCC_DIAG_ON(x)
#endif

/*---------------------------------------------------------------------------*/
/* stop watch (return milliseconds)                                          */

#if defined(HAVE_GETTIMEOFDAY) && defined(HAVE_SYS_TIME_H)
/* use gettimeofday() command. Not in ANSI C! */

#include <sys/time.h>
typedef struct timer {
  struct timeval tv;
  double start;
  double interim;
  double stop;
} TIMER;
#define stopwatch_get_time(tv) \
  ( gettimeofday(&tv, NULL), ((tv).tv_sec * 1.e3 + (tv).tv_usec * 1.e-3) )

#else
/* use clock() command. ANSI C but less accurate */

#include <time.h>
typedef struct timer {
  clock_t tv;
  double start;
  double interim;
  double stop;
} TIMER;
#define stopwatch_get_time(tv) \
  ( (1.e3 * clock()) / CLOCKS_PER_SEC )

#endif

void stopwatch_start(TIMER *t);
double stopwatch_lap(TIMER *t);
double stopwatch_stop(TIMER *t);

void stopwatch_init(void);
void stopwatch_print( FILE *LOG, const char *format, double etime );

/*---------------------------------------------------------------------------*/
/* set alarm when run time exceeds given limit                               */
void set_alarm(FILE *LOG);

/*---------------------------------------------------------------------------*/
/* count number of function evaluations                                      */

/* set and start counter for PDF and similar functions in parameter object */
int start_counter_fcalls( UNUR_PAR *par );
/* IMPORTANT:
 * This function uses global variables.
 * Thus the corresponding parameter/generator object has to be DESTROYED
 * before this routine is called again.
 * In addition, it creates a clone of the distribution object to which
 * the parameter objects points to. 
 * Thus one should run 
 *    stop_counter_fcalls() 
 * immediately after the corresponding parameter/generator object has been
 * destroyed to avoid memory leaks.
 */

int stop_counter_fcalls(void);
/* stop counter for PDF calls and clear memory */

/* reset counter to 0 */
void reset_counter_fcalls(void);

/* get number of PDF evaluations */
int get_counter_pdf(void);
int get_counter_logpdf(void);
int get_counter_cdf(void);

/*---------------------------------------------------------------------------*/
/* print header for test log file                                            */
void print_test_log_header( FILE *LOG, unsigned long seed, int fullcheck );

/*---------------------------------------------------------------------------*/
/* check for invalid NULL pointer, that should not happen in this program */
void abort_if_NULL( FILE *LOG, int line, const void *ptr );

/* compare error code */
int check_errorcode( FILE *LOG, int line, int cherrno );

/* check for expected NULL pointer */
/* int do_check_expected_NULL( FILE *LOG, int line, const void *ptr ); */
int do_check_expected_NULL( FILE *LOG, int line, int is_NULL );
#define check_expected_NULL(LOG,line,ptr) \
   do_check_expected_NULL((LOG),(line),((ptr)==NULL)?1:0 )

/* check for "set failed" */
int check_expected_setfailed( FILE *LOG, int line, int rcode );

/* check for expected zero (int 0) */
int check_expected_zero( FILE *LOG, int line, int k );

/* check for INFINITY */
int check_expected_INFINITY( FILE *LOG, int line, double x );
int check_expected_negINFINITY( FILE *LOG, int line, double x );
int check_expected_INTMAX( FILE *LOG, int line, int k );

/* check for reinit */
int check_expected_reinit( FILE *LOG, int line, int rcode );

/* check for non existing reinit */
int check_expected_no_reinit( FILE *LOG, int line, int rcode );

/* compare sequences generated by generator */
int compare_sequence_gen_start ( FILE *LOG, int line, UNUR_GEN *gen, int sample_size );
int compare_sequence_gen       ( FILE *LOG, int line, UNUR_GEN *gen, int sample_size );
int compare_sequence_par_start ( FILE *LOG, int line, UNUR_PAR *par, int sample_size );
int compare_sequence_par       ( FILE *LOG, int line, UNUR_PAR *par, int sample_size );
int compare_sequence_urng_start( FILE *LOG, int line, int sample_size );

/* free memory used for comparing sequences */
void compare_free_memory( void );

/* print name of distribution */
void print_distr_name( FILE *LOG, const UNUR_DISTR *distr, const char *genid );

/* check p-value of statistical test and print result */
int print_pval( FILE *LOG, UNUR_GEN *gen, const UNUR_DISTR *distr, double pval, int trial, int todo );

/* run chi2 test */
int run_validate_chi2( FILE *LOG, int line, UNUR_GEN *gen, const UNUR_DISTR *distr, int todo );

/* run verify hat test */
int run_validate_verifyhat( FILE *LOG, int line, UNUR_GEN *gen, const UNUR_DISTR *distr, int todo );

/* print result of verify hat test */
int print_verifyhat_result( FILE *LOG, UNUR_GEN *gen, const UNUR_DISTR *distr, int failed, int todo );

/* print result of timings */
void print_timing_results( FILE *LOG, int line, const UNUR_DISTR *distr,
			   double *timing_setup, double *timing_marginal, int n_results );

/* run test for u-error of inversion method and print results */
int run_validate_u_error( FILE *LOG, UNUR_GEN *gen, const UNUR_DISTR *distr,
			  double u_resolution, int samplesize );

/*---------------------------------------------------------------------------*/
