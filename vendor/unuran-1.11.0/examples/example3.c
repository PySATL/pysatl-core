/* ------------------------------------------------------------- */
/* File: example3.c                                              */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;          /* loop variable                            */
  double x;          /* will hold the random number              */

  double fparams[2]; /* array for parameters for distribution    */

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Use a predefined standard distribution:                     */
  /*   Gaussian with mean 2. and standard deviation 0.5.         */
  fparams[0] = 2.;
  fparams[1] = 0.5;
  distr = unur_distr_normal( fparams, 2 );

  /* Choose a method: TDR.                                       */
  par = unur_tdr_new(distr);

  /* Change some of the default parameters.                      */

  /* We want to use T(x)=log(x) for the transformation.          */
  unur_tdr_set_c( par, 0. );

  /* We want to have the variant with immediate acceptance.      */
  unur_tdr_set_variant_ia( par );

  /* We want to use 10 construction points for the setup         */
  unur_tdr_set_cpoints ( par, 10, NULL );

  /* Create the generator object.                                */
  gen = unur_init(par);

  /* Notice that this call has also destroyed the parameter      */
  /* object `par' as a side effect.                              */

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
