/* ------------------------------------------------------------- */
/* File: example_rngstreams.c                                    */
/* ------------------------------------------------------------- */
#ifdef UNURAN_SUPPORTS_RNGSTREAM
/* ------------------------------------------------------------- */
/* This example makes use of the RNGSTREAM library for           */
/* for generating uniform random numbers.                        */
/* (see http://statmath.wu.ac.at/software/RngStreams/)           */
/* To compile this example you must have set                     */
/*   ./configure --with-urng-rngstream                           */
/* (Of course the executable has to be linked against the        */
/* RNGSTREAM library.)                                           */
/* ------------------------------------------------------------- */

/* Include UNURAN header files.                                  */
#include <unuran.h>
#include <unuran_urng_rngstreams.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;          /* loop variable                            */
  double x;          /* will hold the random number              */
  double fparams[2]; /* array for parameters for distribution    */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Declare objects for uniform random number generators.       */
  UNUR_URNG  *urng1, *urng2;    /* uniform generator objects     */

  /* The RNGSTREAMS library sets a package seed.                 */
  unsigned long seed[] = {111u, 222u, 333u, 444u, 555u, 666u};
  RngStream_SetPackageSeed(seed);

  /* RngStreams only:                                            */
  /* Make a object for uniform random number generator.          */
  /* For details see                                             */
  /* http://statmath.wu.ac.at/software/RngStreams/               */
  urng1 = unur_urng_rngstream_new("urng-1");
  if (urng1 == NULL) exit (EXIT_FAILURE);

  /* Use a predefined standard distribution:                     */
  /*   Beta with parameters 2 and 3.                             */
  fparams[0] = 2.;
  fparams[1] = 3.;
  distr = unur_distr_beta( fparams, 2 );

  /* Choose a method: TDR.                                       */
  par = unur_tdr_new(distr);

  /* Set uniform generator in parameter object                   */
  unur_set_urng( par, urng1 );

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

  /* Now we want to switch to a different (independent) stream   */
  /* of uniform random numbers.                                  */
  urng2 = unur_urng_rngstream_new("urng-2");
  if (urng2 == NULL) exit (EXIT_FAILURE);
  unur_chg_urng( gen, urng2 );

  /* ... and sample again.                                       */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  /* We also should destroy the uniform random number generators.*/
  unur_urng_free(urng1);
  unur_urng_free(urng2);

  exit (EXIT_SUCCESS);
} /* end of main() */

/* ------------------------------------------------------------- */
#else
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  printf("You must enable the RNGSTREAM library to run this example!\n\n");
  exit (77);    /* exit code for automake check routines */
}
#endif
/* ------------------------------------------------------------- */

