/* ------------------------------------------------------------- */
/* File: example_mcorr.c                                         */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Example how to generate random correlation matrices           */

/* ------------------------------------------------------------- */

#define dim (4)

int main(void)
{
  int    i,j;
  double M[dim*dim];

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create a distribution object for random correlation matrix  */
  distr = unur_distr_correlation( dim );

  /* Choose a method: MCORR.                                     */
  par = unur_mcorr_new(distr);

  /* Create the generator object.                                */
  gen = unur_init(par);

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
    unur_sample_matr(gen, M);
    for (j=0; j<dim; j++)
      printf("( % f % f % f % f )\n", 
	     M[j*dim+0], M[j*dim+1], M[j*dim+2], M[j*dim+3]);
    printf("\n");
  }
 
  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */
