/* ------------------------------------------------------------- */
/* File: example_vemp.c                                          */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Example how to sample from an empirial continuous             */
/* multivariate distribution.                                    */

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;

  /* 4 data points of dimension 2                                */
  double data[] = { 1. ,1.,       /* 1st data point              */
		    -1.,1.,       /* 2nd data point              */
		    1.,-1.,       /* 3rd data point              */
		    -1.,-1. };    /* 4th data point              */

  double result[2];

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create a distribution object with dimension 2.              */
  distr = unur_distr_cvemp_new( 2 );

  /* Set empirical sample.                                       */
  unur_distr_cvemp_set_data(distr, data, 4); 

  /* Choose a method: VEMPK.                                     */
  par = unur_vempk_new(distr);

  /* Use variance correction.                                    */
  unur_vempk_set_varcor( par, 1 );

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
    unur_sample_vec(gen, result);
    printf("(%f,%f)\n", result[0], result[1]);
  }
 
  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */
