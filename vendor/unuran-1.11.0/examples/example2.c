/* ------------------------------------------------------------- */
/* File: example2.c                                              */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* In this example we build a distribution object from scratch   */
/* and sample from this distribution.                            */
/*                                                               */
/* We use method TDR (Transformed Density Rejection) which       */
/* required a PDF and the derivative of the PDF.                 */

/* ------------------------------------------------------------- */

/* Define the PDF and dPDF of our distribution.                  */
/*                                                               */
/* Our distribution has the PDF                                  */
/*                                                               */
/*          /  1 - x*x  if |x| <= 1                              */
/*  f(x) = <                                                     */
/*          \  0        otherwise                                */
/*                                                               */

/* The PDF of our distribution:                                  */
double mypdf( double x, const UNUR_DISTR *distr )
     /* The second argument (`distr') can be used for parameters */
     /* for the PDF. (We do not use parameters in our example.)  */
{
  if (fabs(x) >= 1.)
    return 0.;
  else
    return (1.-x*x);
} /* end of mypdf() */

/* The derivative of the PDF of our distribution:                */
double mydpdf( double x, const UNUR_DISTR *distr )
{
  if (fabs(x) >= 1.)
    return 0.;
  else
    return (-2.*x);
} /* end of mydpdf() */

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;     /* loop variable                                 */
  double x;     /* will hold the random number                   */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create a new distribution object from scratch.              */
  /* It is a continuous distribution, and we need a PDF and the  */
  /* derivative of the PDF. Moreover we set the domain.          */

  /* Get empty distribution object for a continuous distribution */
  distr = unur_distr_cont_new();

  /* Assign the PDF and dPDF (defined above).                    */
  unur_distr_cont_set_pdf( distr, mypdf );
  unur_distr_cont_set_dpdf( distr, mydpdf );

  /* Set the domain of the distribution (optional for TDR).      */
  unur_distr_cont_set_domain( distr, -1., 1. );

  /* Choose a method: TDR.                                       */
  par = unur_tdr_new(distr);

  /* Now you can change some of the default settings for the     */
  /* parameters of the chosen method. We don't do it here.       */

  /* Create the generator object.                                */
  gen = unur_init(par);

  /* Notice that this call has also destroyed the parameter      */
  /* object `par' as a side effect.                              */

  /* It is important to check if the creation of the generator   */
  /* object was successful. Otherwise `gen' is the NULL pointer  */ 
  /* and would cause a segmentation fault if used for sampling.  */
  if (gen == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }

  /* It is possible to reuse the distribution object to create   */
  /* another generator object. If you do not need it any more,   */
  /* it should be destroyed to free memory.                      */
  unur_distr_free(distr);

  /* Now you can use the generator object `gen' to sample from   */
  /* the distribution. Eg.:                                      */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */


