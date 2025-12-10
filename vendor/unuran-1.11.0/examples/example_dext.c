/* ------------------------------------------------------------- */
/* File: example_dext.c                                          */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* This example shows how an external generator for the          */
/* geometric distribution can be used within the UNURAN          */
/* framework.                                                    */
/*                                                               */
/* Notice, that this example does not provide the simplest       */
/* solution.                                                     */

/* ------------------------------------------------------------- */
/* Initialization routine.                                       */
/*                                                               */
/*   Here we simply read the parameter of the geometric          */
/*   distribution and store it in an array for parameters of     */
/*   the external generator.                                     */
/*   [ Of course we could do this in the sampling routine as     */
/*   and avoid the necessity of this initialization routine. ]   */

int geometric_init (UNUR_GEN *gen)
{ 
  /* Get pointer to parameters of geometric distribution         */
  double *params = unur_dext_get_distrparams(gen);

  /* The parameter is the first entry (see manual)               */
  double p = params[0];

  /* Get array to store this parameter for external generator    */
  double *genpar = unur_dext_get_params(gen, sizeof(double));
  genpar[0] = p;

  /* Executed successfully                                       */
  return UNUR_SUCCESS;
}

/* ------------------------------------------------------------- */
/* Sampling routine.                                             */
/*                                                               */
/*   Contains the code for the external generator.               */

int geometric_sample (UNUR_GEN *gen)
{ 
  /* Get scale parameter                                         */
  double *genpar = unur_dext_get_params(gen,0);
  double p = genpar[0];

  /* Sample a uniformly distributed random number                */
  double U = unur_sample_urng(gen);

  /* Transform into geometrically distributed random variate     */
  return ( (int) (log(U) / log(1.-p)) );
}

/* ------------------------------------------------------------- */

int main(void)
{
  int i;     /* loop variable                                    */
  int K;     /* will hold the random number                      */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Use predefined geometric distribution with parameter 1/10   */
  double fpar[1] = { 0.1 };
  distr = unur_distr_geometric(fpar, 1);

  /* Use method DEXT                                             */
  par = unur_dext_new(distr);

  /* Set initialization and sampling routines.                   */
  unur_dext_set_init(par, geometric_init);
  unur_dext_set_sample(par, geometric_sample);

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
  /* the standard Gaussian distribution.                         */
  /* Eg.:                                                        */
  for (i=0; i<10; i++) {
    K = unur_sample_discr(gen);
    printf("%d\n",K);
  }

  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */

