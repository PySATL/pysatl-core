/* ------------------------------------------------------------- */
/* File: example3_str.c                                              */
/* ------------------------------------------------------------- */
/* String API.                                                   */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;          /* loop variable                            */
  double x;          /* will hold the random number              */

  /* Declare UNURAN generator object.                            */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create the generator object.                                */
  /* Use a predefined standard distribution:                     */
  /*   Gaussian with mean 2. and standard deviation 0.5.         */
  /* Choose a method: TDR with parameters                        */
  /*   c = 0: use T(x)=log(x) for the transformation;            */
  /*   variant "immediate acceptance";                           */
  /*   number of construction points = 10.                       */
  gen = unur_str2gen(
       "normal(2,0.5) & method=tdr; c=0.; variant_ia; cpoints=10");         

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

  /* It is possible with method TDR to truncate the distribution */
  /* for an existing generator object ...                        */
  unur_tdr_chg_truncated( gen, -1., 0. );

  /* ... and sample again.                                       */
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
