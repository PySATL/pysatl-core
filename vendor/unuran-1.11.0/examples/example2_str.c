/* ------------------------------------------------------------- */
/* File: example2_str.c                                          */
/* ------------------------------------------------------------- */
/* String API.                                                   */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* In this example we use a generic distribution object          */
/* and sample from this distribution.                            */
/*                                                               */
/* The PDF of our distribution is given by                       */
/*                                                               */
/*          /  1 - x*x  if |x| <= 1                              */
/*  f(x) = <                                                     */
/*          \  0        otherwise                                */
/*                                                               */
/* We use method TDR (Transformed Density Rejection) which       */
/* required a PDF and the derivative of the PDF.                 */

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;     /* loop variable                                 */
  double x;     /* will hold the random number                   */

  /* Declare UNURAN generator object.                            */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create the generator object.                                */
  /* Use a generic continuous distribution.                      */
  /* Choose a method: TDR.                                       */
  gen = unur_str2gen(
        "distr = cont; pdf=\"1-x*x\"; domain=(-1,1) & method=tdr");

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

  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */


