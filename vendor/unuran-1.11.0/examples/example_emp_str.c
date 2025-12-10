/* ------------------------------------------------------------- */
/* File: example_emp_str.c                                       */
/* ------------------------------------------------------------- */
/* String API.                                                   */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Example how to sample from an empirial continuous univariate  */
/* distribution.                                                 */

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;
  double x;
  
  /* Declare UNURAN generator object.                            */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create the generator object.                                */
  gen = unur_str2gen("distr = cemp; \
                         data=(-0.10, 0.05,-0.50, 0.08, 0.13,   \
                               -0.21,-0.44,-0.43,-0.33,-0.30,   \
                                0.18, 0.20,-0.37,-0.29,-0.90) & \
                      method=empk; smoothing=0.8");

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
