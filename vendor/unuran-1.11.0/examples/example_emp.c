/* ------------------------------------------------------------- */
/* File: example_emp.c                                          */
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
  
  /* data points                                                 */
  double data[15] = { -0.1,  0.05, -0.5,   0.08,  0.13,\
		      -0.21,-0.44, -0.43, -0.33, -0.3, \
		       0.18, 0.2,  -0.37, -0.29, -0.9 };

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create a distribution object and set empirical sample.      */
  distr = unur_distr_cemp_new();
  unur_distr_cemp_set_data(distr, data, 15); 

  /* Choose a method: EMPK.                                      */
  par = unur_empk_new(distr);

  /* Set smooting factor.                                        */
  unur_empk_set_smoothing(par, 0.8);
  
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
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */
