/* ------------------------------------------------------------- */
/* File: example2.c                                              */
/* ------------------------------------------------------------- */

/* Include UNURAN header files.                                  */
#include <unuran.h>
#include <unuran_urng_rngstreams.h>

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;           /* loop variable                           */
  double x; int k;    /* will hold the random number             */

  /* Declare UNU.RAN objects.                                    */
  UNUR_DISTR *distr;             /* distribution object          */
  UNUR_PAR   *par;               /* parameter object             */
  UNUR_GEN  *gen1, *gen2, *gen3; /* generator objects            */

  /* Declare objects for uniform random number generators.       */
  UNUR_URNG  *urng2;       /* uniform RN generator object        */

  /* -- Optional: Set seed for RNGSTREAMS library -------------- */

  /* The RNGSTREAMS library sets a package seed.                 */
  unsigned long seed[] = {111u, 222u, 333u, 444u, 555u, 666u};
  RngStream_SetPackageSeed(seed);

  /* -- Example 1 ---------------------------------------------- */
  /* Beta distribution with shape parameters 2 and 3.            */
  /* Use method 'AUTO' (AUTOmatic).                              */

  /* Create distribution object.                                 */
  {
    double fparams[] = {2., 3.};
    distr = unur_distr_beta( fparams, 2 );
  }

  /* Choose a method: 'AUTO'.                                    */
  par = unur_auto_new(distr);

  /* Create the generator object.                                */
  gen1 = unur_init(par);

  /* It is important to check if the creation of the generator   */
  /* object was successful. Otherwise `gen1' is the NULL pointer */ 
  /* and would cause a segmentation fault if used for sampling.  */
  if (gen1 == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }

  /* It is possible to reuse the distribution object to create   */
  /* another generator object. If you do not need it any more,   */
  /* it should be destroyed to free memory.                      */
  unur_distr_free(distr);

  /* Now you can use the generator object `gen1' to sample from  */
  /* the target distribution. Eg.:                               */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen1);
    printf("%f\n",x);
  }

  /* -- Example 2 ---------------------------------------------- */
  /* Student's t distribution with 3 degrees of freedom.         */
  /* Use method 'TDR' (Transformed Density Rejection) with       */
  /* "immediate acception"                                       */

  /* Create distribution object.                                 */
  {
    double fparams[] = {3.};
    distr = unur_distr_student( fparams, 1 );
  }

  /* Choose a method: 'TDR'.                                     */
  par = unur_tdr_new(distr);
  /* ... and change to immediate acceptance.                     */
  unur_tdr_set_variant_ia(par);

  /* However, this time we use a (new) independent stream of     */
  /* uniformrandom numbers.                                      */
  urng2 = unur_urng_rngstream_new("urng2");
  unur_set_urng( par, urng2 );

  /* Create the generator object.                                */
  gen2 = unur_init(par);
  if (gen2 == NULL) exit (EXIT_FAILURE);

  /* Destroy distribution object. (We do not need it any more.)  */
  unur_distr_free(distr);

  /* Draw a sample.                                              */
  for (i=0; i<10; i++) {
    x = unur_sample_cont(gen2); printf("%f\n",x);
  }

  /* -- Example 3 ---------------------------------------------- */
  /* Discrete distribution with given probability vector.        */
  /* Use method 'DGT' (Discrete Guide Table method).             */

  /* Create distribution object.                                 */
  {
    double probs[] = {0.5, 1.5, 1.0, 0.3};
    distr = unur_distr_discr_new();
    unur_distr_discr_set_pv(distr, probs, 4);
  }

  /* Choose a method: 'DGT'.                                     */
  par = unur_dgt_new(distr);

  /* we use the default URNG again. So there is nothing to do.   */

  /* Create the generator object.                                */
  gen3 = unur_init(par);
  if (gen3 == NULL) exit (EXIT_FAILURE);

  /* Destroy distribution object. (We do not need it any more.)  */
  unur_distr_free(distr);

  /* Draw a sample. Notice that we get integers!                 */
  for (i=0; i<10; i++) {
    k = unur_sample_discr(gen3); printf("%d\n",k);
  }

  /* -- Call destructor ---------------------------------------- */
  /* When generators are not needed any they can be destroyed.   */

  unur_free(gen1);
  unur_free(gen2);  unur_urng_free(urng2);
  unur_free(gen3);

  exit (EXIT_SUCCESS);
} /* end of main() */

/* ------------------------------------------------------------- */
