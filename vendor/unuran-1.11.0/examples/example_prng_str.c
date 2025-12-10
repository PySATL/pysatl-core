/* ------------------------------------------------------------- */
/* File: example_prng_str.c                                      */
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

/* Include UNURAN header files.                                  */
#include <unuran.h>
#include <unuran_urng_prng.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;          /* loop variable                            */
  double x;          /* will hold the random number              */

  /* Declare UNURAN generator object.                            */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Declare objects for uniform random number generators.       */
  UNUR_URNG  *urng1, *urng2;    /* uniform generator objects     */

  /* Create the generator object.                                */
  /* Use a predefined standard distribution:                     */
  /*   Beta with parameters 2 and 3.                             */
  /* Choose a method: TDR.                                       */
  /* Use the Mersenne Twister for unifrom random number          */
  /*   generator (requires PRNG library).                        */
  gen = unur_str2gen("beta(2,3) & method=tdr & urng = mt19937(1237)");

  /* It is important to check if the creation of the generator   */
  /* object was successful. Otherwise `gen' is the NULL pointer  */ 
  /* and would cause a segmentation fault if used for sampling.  */
  if (gen == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }

  /* Now you can use the generator object `gen' to sample from   */
  /* the distribution. Eg.:                                      */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* Now we want to switch to a different uniform random number  */
  /* generator.                                                  */
  /* Now we use an ICG (Inversive Congruental Generator).        */
  urng2 = unur_urng_prng_new("icg(2147483647,1,1,0)");
  if (urng2 == NULL) exit (EXIT_FAILURE);

  /* Change uniform random number generator.                     */
  /* Notice however that we should save the pointer to uniform   */
  /* random number generator in the generator object.            */
  urng1 = unur_chg_urng( gen, urng2 );

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
  printf("You must enable the PRNG library to run this example!\n\n");
  exit (77);    /* exit code for automake check routines */
}
#endif
/* ------------------------------------------------------------- */
