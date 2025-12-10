/* ------------------------------------------------------------- */
/* File: example_anti.c                                          */
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
  double fparams[2]; /* array for parameters for distribution    */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen_normal, *gen_gamma;
                        /* generator objects                     */

  /* Declare objects for uniform random number generators.       */
  UNUR_URNG  *urng1, *urng2;    /* uniform generator objects     */

  /* PRNG only:                                                  */
  /* Make a object for uniform random number generator.          */
  /* For details see http://statmath.wu.ac.at/prng/.             */


  /* The first generator: Gaussian N(2,5) */

  /* uniform generator: We use the Mersenne Twister.             */
  urng1 = unur_urng_prng_new("mt19937(1237)");
  if (urng1 == NULL) exit (EXIT_FAILURE);

  /* UNURAN generator object for N(2,5) */
  fparams[0] = 2.;
  fparams[1] = 5.;
  distr = unur_distr_normal( fparams, 2 );

  /* Choose method TDR with variant PS.                          */
  par = unur_tdr_new( distr );
  unur_tdr_set_variant_ps( par );

  /* Set uniform generator in parameter object.                  */
  unur_set_urng( par, urng1 );

  /* Set auxilliary uniform random number generator.             */
  /* We use the default generator.                               */
  unur_use_urng_aux_default( par );

  /* Alternatively you can create and use your own auxilliary    */
  /* uniform random number generator:                            */
  /*    UNUR_URNG  *urng_aux;                                    */
  /*    urng_aux = unur_urng_prng_new("tt800");                  */
  /*    if (urng_aux == NULL) exit (EXIT_FAILURE);               */
  /*    unur_set_urng_aux( par, urng_aux );                      */                    

  /* Create the generator object.                                */
  gen_normal = unur_init(par);
  if (gen_normal == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }

  /* Destroy distribution object (gen_normal has its own copy).  */
  unur_distr_free(distr);


  /* The second generator: Gamma(4) with antithetic variates.    */

  /* uniform generator: We use the Mersenne Twister.             */
  urng2 = unur_urng_prng_new("anti(mt19937(1237))");
  if (urng2 == NULL) exit (EXIT_FAILURE);

  /* UNURAN generator object for gamma(4) */
  fparams[0] = 4.;
  distr = unur_distr_gamma( fparams, 1 );

  /* Choose method TDR with variant PS.                          */
  par = unur_tdr_new( distr );
  unur_tdr_set_variant_ps( par );

  /* Set uniform generator in parameter object.                  */
  unur_set_urng( par, urng2 );

  /* Set auxilliary uniform random number generator.             */
  /* We use the default generator.                               */
  unur_use_urng_aux_default( par );

  /* Alternatively you can create and use your own auxilliary    */
  /* uniform random number generator (see above).                */
  /* Notice that both generator objects gen_normal and           */
  /* gen_gamma can share the same auxilliary URNG.               */

  /* Create the generator object.                                */
  gen_gamma = unur_init(par);
  if (gen_gamma == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }

  /* Destroy distribution object (gen_normal has its own copy).  */
  unur_distr_free(distr);


  /* Now we can sample pairs of negatively correlated random     */
  /* variates. E.g.:                                             */
  for (i=0; i<10; i++) {
    xn = unur_sample_cont(gen_normal);
    xg = unur_sample_cont(gen_gamma);
    printf("%g, %g\n",xn,xg);
  }


  /* When you do not need the generator objects any more, you    */
  /* can destroy it.                                             */
  unur_free(gen_normal);
  unur_free(gen_gamma);

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
  printf("You must enable the PRNG library to run this example!\n\n");
  exit (77);    /* exit code for automake check routines */
}
#endif
/* ------------------------------------------------------------- */
