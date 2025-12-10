/* ------------------------------------------------------------- */
/* File: example_anti_str.c                                      */
/* ------------------------------------------------------------- */
/* String API.                                                   */
/* ------------------------------------------------------------- */
#ifdef UNURAN_SUPPORTS_PRNG
/* ------------------------------------------------------------- */
/* This example makes use of the PRNG library for generating     */
/* uniform random numbers.                                       */
/* (see http://statmath.wu.ac.at/prng/)                          */
/* To compile this example you must have set                     */
/*   ./configure --with-urng-prng                                */
/* (Of course the executable has to be linked against the        */
/* PRNG library.)                                                */
/* ------------------------------------------------------------- */

/* Example how to sample from two streams of antithetic random   */
/* variates from Gaussian N(2,5) and Gamma(4) distribution, resp.*/

/* ------------------------------------------------------------- */

/* Include UNURAN header files.                                  */
#include <unuran.h>
#include <unuran_urng_prng.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;          /* loop variable                            */
  double xn, xg;     /* will hold the random number              */

  /* Declare UNURAN generator objects.                           */
  UNUR_GEN   *gen_normal, *gen_gamma;

  /* PRNG only:                                                  */
  /* Make a object for uniform random number generator.          */
  /* For details see http://statmath.wu.ac.at/prng/.             */

  /* Create the first generator: Gaussian N(2,5)                 */
  gen_normal = unur_str2gen("normal(2,5) & method=tdr; variant_ps & urng=mt19937(1237)");
  if (gen_normal == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }
  /* Set auxilliary uniform random number generator.             */
  /* We use the default generator.                               */
  unur_chgto_urng_aux_default(gen_normal);

  /* The second generator: Gamma(4) with antithetic variates.    */
  gen_gamma = unur_str2gen("gamma(4) & method=tdr; variant_ps & urng=anti(mt19937(1237))");
  if (gen_gamma == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }
  unur_chgto_urng_aux_default(gen_gamma);


  /* Now we can sample pairs of negatively correlated random     */
  /* variates. E.g.:                                             */
  for (i=0; i<10; i++) {
    xn = unur_sample_cont(gen_normal);
    xg = unur_sample_cont(gen_gamma);
    printf("%g, %g\n",xn,xg);
  }


  /* When you do not need the generator objects any more, you    */
  /* can destroy it.                                             */

  /* But first we have to destroy the uniform random number      */
  /* generators.                                                 */
  unur_urng_free(unur_get_urng(gen_normal));
  unur_urng_free(unur_get_urng(gen_gamma));

  unur_free(gen_normal);
  unur_free(gen_gamma);


  exit (EXIT_SUCCESS);
} /* end of main() */

/* ------------------------------------------------------------- */
#else
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  printf("You must enable the PRNG library to run this example!\n\n");
  exit (77);    /* exit code for automake check routines */
}
#endif
/* ------------------------------------------------------------- */
