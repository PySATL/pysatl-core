/* ------------------------------------------------------------- */
/* File: example_cext.c                                          */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* This example shows how an external generator for the          */
/* exponential distribution with one scale parameter can be      */
/* used within the UNURAN framework.                             */
/*                                                               */
/* Notice, that this example does not provide the simplest       */
/* solution.                                                     */

/* ------------------------------------------------------------- */
/* Initialization routine.                                       */
/*                                                               */
/*   Here we simply read the scale parameter of the exponential  */
/*   distribution and store it in an array for parameters of     */
/*   the external generator.                                     */
/*   [ Of course we could do this in the sampling routine as     */
/*   and avoid the necessity of this initialization routine. ]   */

int exponential_init (UNUR_GEN *gen)
{ 
  /* Get pointer to parameters of exponential distribution       */
  double *params = unur_cext_get_distrparams(gen);

  /* The scale parameter is the first entry (see manual)         */
  double lambda = (params) ? params[0] : 1.;

  /* Get array to store this parameter for external generator    */
  double *genpar = unur_cext_get_params(gen, sizeof(double));
  genpar[0] = lambda;

  /* Executed successfully                                       */
  return UNUR_SUCCESS;
}

/* ------------------------------------------------------------- */
/* Sampling routine.                                             */
/*                                                               */
/*   Contains the code for the external generator.               */

double exponential_sample (UNUR_GEN *gen)
{ 
  /* Get scale parameter                                         */
  double *genpar = unur_cext_get_params(gen,0);
  double lambda = genpar[0];

  /* Sample a uniformly distributed random number                */
  double U = unur_sample_urng(gen);

  /* Transform into exponentially distributed random variate     */
  return ( -log(1. - U) * lambda );
}

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;     /* loop variable                                 */
  double x;     /* will hold the random number                   */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Use predefined exponential distribution with scale param. 2 */
  double fpar[1] = { 2. };
  distr = unur_distr_exponential(fpar, 1);

  /* Use method CEXT                                             */
  par = unur_cext_new(distr);

  /* Set initialization and sampling routines.                   */
  unur_cext_set_init(par, exponential_init);
  unur_cext_set_sample(par, exponential_sample);

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
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* When you do not need the generator object any more, you     */
  /* can destroy it.                                             */
  unur_free(gen);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */

