/* ------------------------------------------------------------- */
/* File: example1_str.c                                          */
/* ------------------------------------------------------------- */
/* String API.                                                   */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;     /* loop variable                                 */
  double x;     /* will hold the random number                   */

  /* Declare UNURAN generator object.                            */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create the generator object.                                */
  /* Use a predefined standard distribution:                     */
  /*   Standard Gaussian distribution.                           */
  /* Choose a method: AROU.                                      */
  /*   For other (suitable) methods replace "arou" with the      */
  /*   respective name.                                          */
  gen = unur_str2gen("normal() & method=arou");         

  /* It is important to check if the creation of the generator   */
  /* object was successful. Otherwise `gen' is the NULL pointer  */ 
  /* and would cause a segmentation fault if used for sampling.  */
  if (gen == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }

  /* Now you can use the generator object `gen' to sample from   */
  /* the standard Gaussian distribution.                         */
  /* Eg.:                                                        */
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

