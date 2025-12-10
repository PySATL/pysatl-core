/* ------------------------------------------------------------- */
/* File: example_discr_str.c                                     */
/* ------------------------------------------------------------- */
/* String API.                                                   */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Example how to sample from a discrete univariate distribution.*/

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;     /* loop variable                                 */

  /* Declare UNURAN generator objects.                           */
  UNUR_GEN   *gen1, *gen2;        /* generator objects           */


  /* First distribution: defined by PMF.                         */
  gen1 = unur_str2gen("geometric(0.3); mode=0 & method=dari");

  /* It is important to check if the creation of the generator   */
  /* object was successful. Otherwise `gen' is the NULL pointer  */ 
  /* and would cause a segmentation fault if used for sampling.  */
  if (gen1 == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }
  
  /* Second distribution: defined by (finite) PV.                */
  gen2 = unur_str2gen(
             "distr=discr; pv=(1,2,3,4,5,6,7,8,4,3) & method=dgt");
  if (gen2 == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }
  
  /* print some random integers                                  */
  for (i=0; i<10; i++){
    printf("number %d: %d\n", i*2,   unur_sample_discr(gen1) );
    printf("number %d: %d\n", i*2+1, unur_sample_discr(gen2) );
  }

  /* Destroy all objects.                                        */
  unur_free(gen1);
  unur_free(gen2);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */
