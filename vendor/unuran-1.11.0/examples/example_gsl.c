/* ------------------------------------------------------------- */
/* File: example_gsl.c                                           */
/* ------------------------------------------------------------- */
#ifdef UNURAN_SUPPORTS_GSL
/* ------------------------------------------------------------- */
/* This example makes use of the GSL library for generating      */
/* uniform random numbers.                                       */
/* (see http://www.gnu.org/software/gsl/)                        */
/* To compile this example you must have set                     */
/*   ./configure --with-urng-gsl                                 */
/* (Of course the executable has to be linked against the        */
/* GSL library.)                                                 */
/* ------------------------------------------------------------- */

/* Include UNURAN header files.                                  */
#include <unuran.h>
#include <unuran_urng_gsl.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;          /* loop variable                            */
  double x;          /* will hold the random number              */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Declare objects for uniform random number generators.       */
  UNUR_URNG  *urng;     /* uniform generator objects             */

  /* GNU Scientific Library only:                                */
  /* Make a object for uniform random number generator.          */
  urng = unur_urng_gsl_new(gsl_rng_mt19937);
  if (urng == NULL) exit (EXIT_FAILURE);

  /* Create a generator object using this URNG */
  distr = unur_distr_normal( NULL, 0 );
  par = unur_tdr_new(distr);
  unur_set_urng( par, urng );
  gen = unur_init(par);
  if (gen == NULL)  exit (EXIT_FAILURE);
  unur_distr_free(distr);

  /* Now you can use the generator object `gen' to sample from   */
  /* the distribution. Eg.:                                      */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* Destroy objects */
  unur_free(gen);
  unur_urng_free(urng);

  exit (EXIT_SUCCESS);
} /* end of main() */

/* ------------------------------------------------------------- */
#else
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  printf("You must enable the GSL to run this example!\n\n");
  exit (77);    /* exit code for automake check routines */
}
#endif
/* ------------------------------------------------------------- */

