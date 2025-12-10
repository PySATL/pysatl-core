/* ------------------------------------------------------------- */
/* File: example_reinit.c                                        */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* In this example we show how the parameters of the underlying  */
/* distribution can be changed for an existing generator object. */

/* We use the GIG distribution with method CSTD.                 */

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;     /* loop variable                                 */
  double x;     /* will hold the random number                   */

  /* Parameters of distribution.                                 */ 
  double dparam[3] = {0.5, 1., 5.};

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;    /* distribution object                   */
  UNUR_PAR   *par;      /* parameter object                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Create initial GIG distribution object                      */
  distr = unur_distr_gig(dparam, 3);

  /* Choose a method: CSTD.                                      */
  par = unur_cstd_new(distr);

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

  /* It is possible for method CSTD to change the parameters of  */
  /* underlying distribution. However, we have to extract to     */
  /* pointer to the distribution. Be carefull with this pointer! */
  distr = unur_get_distr(gen);

  /* Change the parameter(s).                                    */
  dparam[2] = 0.001;
  unur_distr_cont_set_pdfparams(distr,dparam,3);

  /* Do not forget to reinitialize the generator object.         */
  /* Check the return code.                                      */
  /* (and try to find a better error handling)                   */
  if (unur_reinit(gen) != UNUR_SUCCESS) {
    fprintf(stderr, "ERROR: cannot reinitialize generator object\n");
    exit (EXIT_FAILURE);
  }

  /* Draw a new sample.                                          */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen);
    printf("%f\n",x);
  }

  /* Changing parameters can be repeated.                        */
  dparam[2] = 1000;
  unur_distr_cont_set_pdfparams(distr,dparam,3);
  if (unur_reinit(gen) != UNUR_SUCCESS) {
    fprintf(stderr, "ERROR: cannot reinitialize generator object\n");
    exit (EXIT_FAILURE);
  }
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


