/***************************************************************************** 
 *                                                                           * 
 *          UNURAN -- Universal Non-Uniform Random number generator          * 
 *                                                                           * 
 *****************************************************************************
 *                                                                           *
 *   FILE:      test_functionparser                                          *
 *                                                                           *
 *   Compare evaluation result of function parser with those of              *
 *   Mathematica(TM)                                                         *
 *                                                                           *
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
/*  #define DEBUG 1 */                                                         
/*---------------------------------------------------------------------------*/
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <config.h>
#include <unuran.h>

/* macros and functions for handling floats */
#include <src/utils/unur_fp_source.h>
#include <src/utils/unur_fp_const_source.h>

/*---------------------------------------------------------------------------*/

/* name of data file */
static const char datafile[] = "t_functionparser.data";

/* number of failed tests */
static int n_failed = 0;    

/* file handles */
FILE *DATA;            /* file with data    */
FILE *UNURANLOG;       /* unuran log file   */
FILE *TESTLOG;         /* test log file     */

/*---------------------------------------------------------------------------*/

static int _unur_test_function (void);

/*---------------------------------------------------------------------------*/

int main(void)                                                                     
{                                                                              
  /* open log file for unuran and set output stream for unuran messages */    
  if ( (UNURANLOG = fopen( "t_functionparser_unuran.log","w" )) == NULL )
    exit (EXIT_FAILURE);                                           
  unur_set_stream( UNURANLOG );
                                                                               
  /* open log file for testing */                                             
  if ( (TESTLOG = fopen( "t_functionparser_test.log","w" )) == NULL )
    exit (EXIT_FAILURE);                                             
                                                                               
  /* write header into log file */                                            
  {                                                                           
    time_t started;                                                          
    fprintf(TESTLOG,"\nUNU.RAN - Universal Non-Uniform RANdom number generator\n\n");
    if (time( &started ) != -1)                                              
      fprintf(TESTLOG,"%s",ctime(&started));                              
    fprintf(TESTLOG,"\n=======================================================\n\n");
    /*      fprintf(TESTLOG,"(Search for string \"data\" to find new section.)\n\n"); */
  }                                                                           

  /* open data file */
  if ( (DATA = fopen( datafile,"r" )) == NULL ) {
    printf("ERROR: could not open file %s \n", datafile);
    fprintf(TESTLOG,"ERROR: could not open file %s \n", datafile);
    exit (77);  /* ignore this error */                                             
  }

  while (_unur_test_function()==TRUE);

  /* close files */
  fclose (UNURANLOG);
  fclose (TESTLOG);
  fclose (DATA);
                               
  /* end */
  if (n_failed > 0) {
    printf("[function parser --> failed]\n");
    exit (EXIT_FAILURE);
  }
  else {
    printf("[function parser --> ok]\n");
    exit (EXIT_SUCCESS);
  }
} /* end of main() */                                                                              

/*---------------------------------------------------------------------------*/

int
_unur_test_function (void)
{
#define BUFSIZE      1024       /* size of line buffer */

  char buffer[BUFSIZE];         /* buffer for reading line */
  char *ptr_buffer;             /* pointer into buffer */
  
  char fstr[BUFSIZE];           /* function string */
  
  double x;                     /* argument x */
  double fx_exp, dfx_exp;       /* expected values for f(x) and f'(x) */
  double fx_obs, dfx_obs;       /* observed values for f(x) and f'(x) */
  double fx_rep, dfx_rep;       /* observed values for f(x) and f'(x) 
				   for reparsed string                */
  double diff;
  char *repstr = NULL;          /* string generated from parsed tree  */
  
  UNUR_DISTR *distr, *rep;
  
  int failed = 0;
  
  /* find next function string */
  while (1) {
    /* read next line */
    fgets(buffer, BUFSIZE, DATA);
    if (feof(DATA)) return FALSE;
    if (strncmp(buffer,"function=",8)==0) break;
  }
  
  /* get function string */
  strcpy( fstr, buffer+9 );
  /* remove newline character */
  fstr[strlen(fstr)-1] = '\0';
  
  /* print info into log file */
  fprintf(TESTLOG,"function = \"%s\"\n", fstr);
  
  /* make distribution object with given function as PDF */
  distr = unur_distr_cont_new();
  if (unur_distr_cont_set_pdfstr(distr,fstr)!=UNUR_SUCCESS) {
    printf("ERROR: syntax error in \"%s\"\n", fstr);
    fprintf(TESTLOG,"ERROR: syntax error in \"%s\"\n", fstr);
    exit (EXIT_FAILURE);                                             
  }
  
  /* reparse function string and make duplicate distribution object */
  repstr = unur_distr_cont_get_pdfstr(distr);
  fprintf(TESTLOG,"parsed   = \"%s\"\n", repstr);
  rep = unur_distr_cont_new();
  if (unur_distr_cont_set_pdfstr(rep,repstr)!=UNUR_SUCCESS) {
    printf("ERROR: syntax error in \"%s\"\n", repstr);
    fprintf(TESTLOG,"ERROR: syntax error in reparsed string \"%s\"\n", repstr);
    exit (EXIT_FAILURE);                                             
  }
  
  /* read all the data */
  while (1) {
    /* read next line */
    fgets(buffer, BUFSIZE, DATA);
    if (feof(DATA)) break;

    /* stop if blank line */
    if (isspace(buffer[0]))
      break;
    
    /* read x and expected values for f(x) and f'(x) */
    ptr_buffer = buffer;
    x = strtod( ptr_buffer, &ptr_buffer );
    fx_exp = strtod( ptr_buffer, &ptr_buffer );
    dfx_exp = strtod( ptr_buffer, &ptr_buffer );
    
    /* compute function values for function tree */
    fx_obs = unur_distr_cont_eval_pdf(x,distr);
    dfx_obs = unur_distr_cont_eval_dpdf(x,distr);

    /* compute function values for reparsed function tree */
    fx_rep = unur_distr_cont_eval_pdf(x,rep);
    dfx_rep = unur_distr_cont_eval_dpdf(x,rep);

    /* compare */
    if (!(_unur_FP_approx(fx_exp,fx_obs))) {
      ++failed;
      diff = fx_obs - fx_exp;
      fprintf(TESTLOG,"[fx]\tx = %g:\t(exp) = %g\t(obs) = %g\tdiff = %g  --> error\n",
	      x,fx_exp,fx_obs,diff);
    }
    if (!(_unur_FP_approx(fx_obs,fx_rep))) {
      ++failed;
      diff = fx_rep - fx_obs;
      fprintf(TESTLOG,"[fx]\tx = %g:\t(obs) = %g\t(rep) = %g\tdiff = %g  --> error\n",
	      x,fx_obs,fx_rep,diff);
    }
    if (!(_unur_FP_approx(dfx_exp,dfx_obs))) {
      ++failed;
      diff = dfx_obs - dfx_exp;
      fprintf(TESTLOG,"[dfx]\tx = %g:\t(exp) = %g\t(obs) = %g\tdiff = %g  --> error\n",
	      x,dfx_exp,dfx_obs,diff);
    }
    if (!(_unur_FP_approx(dfx_obs,dfx_rep))) {
      ++failed;
      diff = dfx_rep - dfx_obs;
      fprintf(TESTLOG,"[dfx]\tx = %g:\t(obs) = %g\t(rep) = %g\tdiff = %g  --> error\n",
	      x,dfx_obs,dfx_rep,diff);
    }
  }

  /* free memory */
  unur_distr_free(distr);
  unur_distr_free(rep);
  if (repstr) free(repstr);

  /* write result */
  if (failed) {
    fprintf(TESTLOG,"\t--> FAILED\n\n");
    ++n_failed;
  }
  else {
    fprintf(TESTLOG,"\t--> OK\n\n");
  }

  return TRUE;
#undef BUFSIZE
} /* end of _unur_test_function() */

/*---------------------------------------------------------------------------*/
