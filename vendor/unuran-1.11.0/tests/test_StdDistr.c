/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: test_StdDistr.c                                                   *
 *                                                                           *
 *   Compare CDF, PDF and derivatives of PDF of varios distributions         *
 *   with Mathematica(TM) output.                                            *
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

/* #define DEBUG 1 */

/*---------------------------------------------------------------------------*/
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <config.h>
#include <unuran.h>

/*---------------------------------------------------------------------------*/

/* PDF test: maximal difference allowed (relative to maximal value) */
#define MAX_REL_DIFF 1.e-10

/* mode test: */
#define EPSX 1.e-5     /* stepsize */
#define EPSY 1.e-12    /* maximal allowed relative error */

/*---------------------------------------------------------------------------*/

/* name of data file */
static const char datafile[] = "t_StdDistr.data";

/* file handles */
static FILE *DATA;            /* file with data    */
static FILE *UNURANLOG;       /* unuran log file   */
static FILE *TESTLOG;         /* test log file     */

/*---------------------------------------------------------------------------*/

static int _unur_test_StdDistr (int *n_failed);
/* main loop for testing standard distributions */

static int test_cdf_pdf( UNUR_DISTR *distr, char *datafile );
/* compare CDF, PDF and dPDF for continuous univariate distributions         */

static int modetest_cont( UNUR_DISTR *distr);
static int modetest_discr( UNUR_DISTR *distr);
/* test mode of distribution */

/*---------------------------------------------------------------------------*/

#define MAX(a,b) ( ((a) < (b)) ? (b) : (a))
#define MIN(a,b) ( ((a) > (b)) ? (b) : (a))

/*---------------------------------------------------------------------------*/

int main()                                                                     
{                                                                              
  /* number of failed tests */
  int n_failed = 0;    

  /* open log file for unuran and set output stream for unuran messages */    
  if ( (UNURANLOG = fopen( "t_StdDistr_unuran.log","w" )) == NULL )
    exit (EXIT_FAILURE);
  unur_set_stream( UNURANLOG );
  
  /* open log file for testing */                                             
  if ( (TESTLOG = fopen( "t_StdDistr_test.log","w" )) == NULL )
    exit (EXIT_FAILURE);
  
  /* write header into log file */                                            
  {                                                                           
    time_t started;                                                          
    fprintf(TESTLOG,"\nUNU.RAN - Universal Non-Uniform RANdom number generator\n\n");
    if (time( &started ) != -1)                                              
      fprintf(TESTLOG,"%s",ctime(&started));                              
    fprintf(TESTLOG,"\n=======================================================\n\n");
    fprintf(TESTLOG,"(Search for string \"distr=\" to find new section.)\n\n");
  }                                                                           
  
  /* open data file */
  if ( (DATA = fopen( datafile,"r" )) == NULL ) {
    printf("ERROR: could not open file %s \n", datafile);
    fprintf(TESTLOG,"ERROR: could not open file %s \n", datafile);
    exit (77);  /* ignore this error */                                             
  }
  
  /* run tests on all distributions */
  while (_unur_test_StdDistr(&n_failed)==TRUE);
  
  /* close files */
  fclose (UNURANLOG);
  fclose (TESTLOG);
  fclose (DATA);
  
  /* end */
  if (n_failed > 0) {
    printf("[StdDistr --> failed]\n");
    exit (EXIT_FAILURE);
  }
  else {
    printf("[StdDistr --> ok]\n");
    exit (EXIT_SUCCESS);
  }
} /* end of main() */                                                                              

/*---------------------------------------------------------------------------*/

int
_unur_test_StdDistr (int *n_failed)
{
#define BUFSIZE      1024       /* size of line buffer */

  char buffer[BUFSIZE];         /* buffer for reading line */
  char dstr[BUFSIZE];           /* distribution string */

  UNUR_DISTR *distr;                     /* distribution object             */

  /* find next function string */
  while (1) {
    /* read next line */
    fgets(buffer, BUFSIZE, DATA);
    if (feof(DATA)) return FALSE;
    if (strncmp(buffer,"distr=",6)==0) break;
  }

  /* get function string */
  strcpy( dstr, buffer+6 );
  /* remove newline character */
  dstr[strlen(dstr)-1] = '\0';

  /* make distribution object with given function as PDF */
  if ( (distr = unur_str2distr(dstr)) == NULL ) {
    printf("ERROR: syntax error in \"%s\"\n", dstr);
    fprintf(TESTLOG,"ERROR: syntax error in \"%s\"\n", dstr);
    exit (EXIT_FAILURE);
  }

  /* now run test */
  if (test_cdf_pdf( distr,dstr ) == UNUR_FAILURE)
    (*n_failed)++;

  /* free memory */
  unur_distr_free(distr);

  return TRUE;

#undef BUFSIZE
} /* end of _unur_test_StdDistr() */

/*---------------------------------------------------------------------------*/

int
test_cdf_pdf( UNUR_DISTR *distr, char *distrAPI )
     /*----------------------------------------------------------------------*/
     /* test CDF, PDF and derivative of PDF by comparing to data             */
     /* read from file created by Mathematica.                               */
     /*                                                                      */
     /* the ordering in this file is:                                        */
     /*   n   par[1] par[2] ... par[n]   x   CDF   PDF   dPDF                */
     /*                                                                      */
     /* parameters:                                                          */
     /*   distr    ... pointer to distribution object                        */
     /*   distrAPI ... string that contains name for distribution            */
     /*                                                                      */
     /* return:                                                              */
     /*   UNUR_SUCCESS  ... if test was successful                           */
     /*   UNUR_FAILURE  ... failed (difference too large)                    */
     /*   UNUR_ERR_NULL ... could not run test                               */
     /*----------------------------------------------------------------------*/
{
#define BUFSIZE      1024       /* size of line buffer */
#define MAX_FPARAMS  10         /* maximal number of parameters for distribution */

  char buffer[BUFSIZE];         /* buffer for reading line */
  char *ptr_buffer;             /* pointer into buffer */

  int n_fparams;                /* number of parameters for distribution */
  double fparams[10];           /* array for parameters for distribution */
  double x;                     /* double argument */
  int k = 0;                    /* integer argument */
  double CDF_e, PDF_e, dPDF_e;  /* expected values for CDF, PDF and derivative of PDF at x */
  double CDF_o, PDF_o, dPDF_o;  /* observed values for CDF, PDF and derivative of PDF at x */
  double CDF_d, PDF_d, dPDF_d;  /* differences between observed and expected values */

  double CDF_md, PDF_md, dPDF_md;  /* maximal difference (absolute) */
  double CDF_me, PDF_me, dPDF_me;  /* maximal expected absolute values */

  int have_CDF, have_PDF, have_dPDF, have_upd_pdfarea;

  int is_DISCR;                 /* 1 if discrete distribution, 0 otherwise */
  int n_failed = 0;             /* number of failed tests */

  const char *dname;            /* name of distribution */

  int i;

  /* discrete distribution ? */
  is_DISCR = unur_distr_is_discr( distr );

  /* name of distribution */
  dname = unur_distr_get_name(distr);

  /* initialize */
  CDF_md = PDF_md = dPDF_md = 0.;
  CDF_me = PDF_me = dPDF_me = 0.;

  /* check existence of CDF, PDF and dPDF */
  if (is_DISCR==TRUE) {
    have_CDF  = (unur_distr_discr_get_cdf(distr)) ? TRUE : FALSE;
    have_PDF  = (unur_distr_discr_get_pmf(distr)) ? TRUE : FALSE;
    have_dPDF = FALSE;
  }
  else { /* is_CONT */
    have_CDF  = (unur_distr_cont_get_cdf(distr))  ? TRUE : FALSE;
    have_PDF  = (unur_distr_cont_get_pdf(distr))  ? TRUE : FALSE;
    have_dPDF = (unur_distr_cont_get_dpdf(distr)) ? TRUE : FALSE;
  }

  /* check whether unur_distr_cont_upd_pdfarea() works */
  if (is_DISCR==TRUE)
    have_upd_pdfarea = (unur_distr_discr_upd_pmfsum(distr)==UNUR_SUCCESS) ? TRUE : FALSE;
  else  /* is_CONT */
    have_upd_pdfarea = (unur_distr_cont_upd_pdfarea(distr)==UNUR_SUCCESS) ? TRUE : FALSE;

  if (have_upd_pdfarea==FALSE) {
    /* if we cannot update the area below the PDF, then the
       given PDF is not a "real" PDF, i.e. not normalized */
    have_PDF = FALSE;
    have_dPDF = FALSE;
  }

  /* print info into log file */
  fprintf(TESTLOG,"%s: distr= %s ...\n", dname, distrAPI);

  if (have_CDF==FALSE)
    fprintf(TESTLOG,"%s: no CDF!\n", dname);
  if (have_PDF==FALSE)
    fprintf(TESTLOG,"%s: no PDF!\n", dname);
  if (have_dPDF==FALSE)
    fprintf(TESTLOG,"%s: no dPDF!\n", dname);

  /* read data file */
  while (1) {

    /* read next line */
    fgets(buffer, BUFSIZE-1, DATA);
    if (feof(DATA) || isspace(buffer[0])) break;
    
    ptr_buffer = buffer;

    /* get number of parameters */
    n_fparams = strtol(buffer, &ptr_buffer, 10);
    if (n_fparams < 0 || n_fparams >= MAX_FPARAMS) {
      printf("%s: ERROR: invalid number of parameters for distribution: %d \n", dname,n_fparams);
      fprintf(TESTLOG,"%s: ERROR: invalid number of parameters for distribution: %d \n", dname,n_fparams);
      return UNUR_ERR_NULL;
    }

    /* read parameters */
    for (i=0; i<n_fparams; i++) {
      fparams[i] = strtod( ptr_buffer, &ptr_buffer );
    }

    /* read argument */
    x = strtod( ptr_buffer, &ptr_buffer );
    if (is_DISCR) 
      /* integer required, round */
      k = (int)(x+0.5);

    /* read CDF */
    CDF_e = strtod( ptr_buffer, &ptr_buffer );

    /* read PDF */
    PDF_e = strtod( ptr_buffer, &ptr_buffer );

    /* read dPDF */
    dPDF_e = strtod( ptr_buffer, &ptr_buffer );

#ifdef DEBUG
    /* print data */
    fprintf(TESTLOG, "%s: n_fparams = %d\n", dname, n_fparams);

    /* print parameters */
    for (i=0; i<n_fparams; i++)
      fprintf(TESTLOG, "%s: \t%d: %g\n", dname, i, fparams[i]);

    /* print argument x (or k) */
    if (is_DISCR)
      fprintf(TESTLOG, "%s: expected k = %d:\t", dname, k);
    else  /* is_CONT */
      fprintf(TESTLOG, "%s: expected x = %g:\t", dname, x);

    /* print CDF, PDF and derivative of PDF at x */
    fprintf(TESTLOG, "%g, %g, %g\n", CDF_e, PDF_e, dPDF_e);
#endif

    if (is_DISCR) {
      /* set parameters for distribution */
      unur_distr_discr_set_pmfparams(distr,fparams,n_fparams);
      if (have_upd_pdfarea) unur_distr_discr_upd_pmfsum(distr); 
      
      /* compute CDF, PDF and derivative of PDF at x */
      CDF_o = (have_CDF) ? unur_distr_discr_eval_cdf(x, distr) : 0.;
      PDF_o = (have_PDF) ? unur_distr_discr_eval_pmf(x, distr) : 0.;
      dPDF_o = 0.;
    }

    else { /* is_CONT */
      /* set parameters for distribution */
      unur_distr_cont_set_pdfparams(distr,fparams,n_fparams);
      if (have_upd_pdfarea) unur_distr_cont_upd_pdfarea(distr); 
      
      /* compute CDF, PDF and derivative of PDF at x */
      CDF_o = (have_CDF) ? unur_distr_cont_eval_cdf(x, distr) : 0.;
      PDF_o = (have_PDF) ? unur_distr_cont_eval_pdf(x, distr) : 0.;
      dPDF_o = (have_dPDF) ? unur_distr_cont_eval_dpdf(x, distr) : 0.;
    }

    /* compute differnces */
    CDF_d = CDF_o - CDF_e;
    PDF_d = PDF_o - PDF_e;
    dPDF_d = dPDF_o - dPDF_e;

#ifdef DEBUG
    /* print argument x (or k) */
    if (is_DISCR)
      fprintf(TESTLOG, "%s: observed k = %d:\t", dname, k);
    else  /* is_CONT */
      fprintf(TESTLOG, "%s: observed x = %g:\t", dname, x);

    /* print CDF, PDF and derivative of PDF at x */
    fprintf(TESTLOG, "%g, %g, %g\n",CDF_o,PDF_o,dPDF_o);

    /* print differences */
    fprintf(TESTLOG, "%s: diff     x = %g:\t", dname,x);
    fprintf(TESTLOG, "%g, %g, %g\n",CDF_d, PDF_d, dPDF_d);
    fprintf(TESTLOG, "%s:\n", dname);
#endif
    
    /* absolute values */
    CDF_d = fabs(CDF_d);
    PDF_d = fabs(PDF_d);
    dPDF_d = fabs(dPDF_d);

    CDF_e = fabs(CDF_e);
    PDF_e = fabs(PDF_e);
    dPDF_e = fabs(dPDF_e);

    /* maximal differences */
    if (CDF_d > CDF_md)   CDF_md = CDF_d;
    if (PDF_d > PDF_md)   PDF_md = PDF_d;
    if (dPDF_d > dPDF_md) dPDF_md = dPDF_d;

    /* maximal values */
    if (CDF_e > CDF_me)   CDF_me = CDF_e;
    if (PDF_e > PDF_me)   PDF_me = PDF_e;
    if (dPDF_e > dPDF_me) dPDF_me = dPDF_e;

    /* test mode of distribution */
    if (is_DISCR) {
      if (modetest_discr(distr) == UNUR_FAILURE)
	++n_failed;
    }
    else { /* is_CONT */
      if (modetest_cont(distr) == UNUR_FAILURE)
	++n_failed;
    }

  }

  /* print info on screen */
  fprintf(TESTLOG, "%s: \tmaximal difference:\n", dname);

  if (have_CDF) {
    fprintf(TESTLOG,"%s: \t\tCDF  = %g ... ", dname,CDF_md);
    if (CDF_md > MAX_REL_DIFF * CDF_me) {
      fprintf(TESTLOG, "failed!!\n");
      ++n_failed;
    }
    else
      fprintf(TESTLOG, "ok\n");
  }

  if (have_PDF) {
    fprintf(TESTLOG,"%s: \t\tPDF  = %g (rel = %g) ... ", dname,PDF_md,PDF_md/PDF_me);
    if (PDF_md > MAX_REL_DIFF * PDF_me) {
      fprintf(TESTLOG, "failed!!\n");
      ++n_failed;
    }
    else
      fprintf(TESTLOG, "ok\n");
  }

  if (have_dPDF) {
    fprintf(TESTLOG,"%s: \t\tdPDF = %g (rel = %g) ... ", dname,dPDF_md,dPDF_md/dPDF_me);
    if (dPDF_md > MAX_REL_DIFF * dPDF_me) {
      fprintf(TESTLOG, "failed!!\n");
      ++n_failed;
    }
    else
      fprintf(TESTLOG, "ok\n");
  }

  fprintf(TESTLOG, "\n----------------------------------------\n\n");

  /* print result on screen */
  printf("%-17s ... ", dname );
  if (n_failed > 0) {
    printf("failed!!\n");
    return UNUR_FAILURE;
  }
  else {
    printf("ok\n");
    return UNUR_SUCCESS;
  }

#undef BUFSIZE
#undef MAX_FPARAMS
} /* end of test_cdf_pdf() */

/*---------------------------------------------------------------------------*/

int modetest_cont( UNUR_DISTR *distr)
     /*----------------------------------------------------------------------*/
     /* Tests whether the mode of continuous distribution is correct.        */
     /*                                                                      */
     /* parameters:                                                          */
     /*   distr    ... pointer to distribution object                        */
     /*                                                                      */
     /* return:                                                              */
     /*   UNUR_SUCCESS  ... if test was successful                           */
     /*   UNUR_FAILURE  ... failed (difference too large)                    */
     /*   UNUR_ERR_NULL ... could not run test                               */
     /*----------------------------------------------------------------------*/
{
  double m, fm;           /* mode and value of PDF at mode                   */
  double x, fx;           /* argument x and value of PDF at x                */
  double domain[2];       /* boundaries of the domain of distribution        */
  const char *dname;      /* name of distribution                            */
  int n_failed = 0;       /* number of failed tests                          */
  int i;                  /* loop variable                                   */

  /* name of distribution */
  dname = unur_distr_get_name(distr);

  /* get mode */
  if (unur_distr_cont_upd_mode(distr)!=UNUR_SUCCESS ) {
    /* cannot read mode */
    printf("%s: ERROR: cannot get mode\n", dname);
    fprintf(TESTLOG,"%s: ERROR: cannot get mode\n", dname);
    return UNUR_ERR_NULL;
  }    
  m = unur_distr_cont_get_mode(distr); 

  /* Correct possible problems if the mode m is on the boundary of domain */
  unur_distr_cont_get_domain(distr,domain, domain+1); 
  if(domain[1] - m < 1.e-11) 
    m = MIN( m-EPSX/2., m*(m<0?(1.+EPSX/2.):1.-EPSX/2.) );
  if(m - domain[0] < 1.e-11)
    m = MAX( m+EPSX/2., m*(m>0?(1.+EPSX/2.):1.-EPSX/2.) );

  /* evaluate PDF at mode */
  fm = unur_distr_cont_eval_pdf(m, distr);

#ifdef DEBUG
  fprintf(TESTLOG,"%s: mode: m = %.20g, f(m) = %.20g\n", dname,m,fm);
#endif

  /* test PDF left and right of mode */
  for (i=0; i<2; i++) {

    if (i==0)
      /* right of mode */
      x = MAX( m+EPSX, m*(m>0?(1.+EPSX):1.-EPSX) );

    else
      /* left of mode */
      x = MIN( m-EPSX, m*(m<0?(1.+EPSX):1.-EPSX) );

    /* evaluate PDF */
    fx = unur_distr_cont_eval_pdf(x, distr);

#ifdef DEBUG
    fprintf(TESTLOG,"%s:       x = %.20g, f(x) = %.20g", dname,x,fx);
#endif

    if(fm * (1.+EPSY) < fx) {
#ifdef DEBUG
      fprintf(TESTLOG," ... failed! f(mode) not maximal!");
#endif
      ++n_failed;
    }

#ifdef DEBUG
    fprintf(TESTLOG,"\n");
#endif
  }

#ifdef DEBUG
    fprintf(TESTLOG,"%s:\n",dname);
#endif

  /* end */
  return ((n_failed > 0) ? UNUR_FAILURE : UNUR_SUCCESS);

} /* end of modetest_cont() */

/*---------------------------------------------------------------------------*/

int modetest_discr( UNUR_DISTR *distr)
     /*----------------------------------------------------------------------*/
     /* Tests whether the mode of discr distribution is correct.             */
     /*                                                                      */
     /* parameters:                                                          */
     /*   distr    ... pointer to distribution object                        */
     /*                                                                      */
     /* return:                                                              */
     /*   UNUR_SUCCESS  ... if test was successful                           */
     /*   UNUR_FAILURE  ... failed (difference too large)                    */
     /*   UNUR_ERR_NULL ... could not run test                               */
     /*----------------------------------------------------------------------*/
{
  int m, x;               /* mode and other argument for PDF                 */
  double fm, fx;          /* value of PDF at m and x                         */
  const char *dname;      /* name of distribution                            */
  int n_failed = 0;       /* number of failed tests                          */
  int i;                  /* loop variable                                   */

  /* name of distribution */
  dname = unur_distr_get_name(distr);

  /* get mode */
  if (unur_distr_discr_upd_mode(distr)!=UNUR_SUCCESS ) {
    /* cannot read mode */
    printf("%s: ERROR: cannot get mode\n", dname);
    fprintf(TESTLOG,"%s: ERROR: cannot get mode\n", dname);
    return UNUR_ERR_NULL;
  }    
  m = unur_distr_discr_get_mode(distr); 

  /* evaluate PMF at mode */
  fm = unur_distr_discr_eval_pmf(m, distr);

#ifdef DEBUG
  fprintf(TESTLOG,"%s: mode: m = %d, f(m) = %.20g\n", dname,m,fm);
#endif

  /* test PMF left and right of mode */
  for (i=-1; i<2; i+=2 ) {
    
    /* left and right of mode */
    x = m + i;

    /* evaluate PMF */
    fx = unur_distr_discr_eval_pmf(x, distr);

#ifdef DEBUG
    fprintf(TESTLOG,"%s:       x = %d, f(x) = %.20g", dname,x,fx);
#endif

    if(fm * (1.+EPSY) < fx) {
#ifdef DEBUG
      fprintf(TESTLOG," ... failed! f(mode) not maximal!");
#endif
      ++n_failed;
    }

#ifdef DEBUG
    fprintf(TESTLOG,"\n");
#endif
  }

#ifdef DEBUG
    fprintf(TESTLOG,"%s:\n",dname);
#endif

  /* end */
  return ((n_failed > 0) ? UNUR_FAILURE : UNUR_SUCCESS);

} /* end of modetest_cont() */

/*---------------------------------------------------------------------------*/

