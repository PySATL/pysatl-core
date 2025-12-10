/* ------------------------------------------------------------- */
/* File: example_mixt_inv.c                                      */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */
/* Mixture of truncated Gaussian on (-INFINITY,0] and            */
/* Exponential distribution on [0,INFINITY)                      */
/* ------------------------------------------------------------- */

int main(void)
{
  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr;   /* distribution object                    */
  UNUR_PAR   *par;     /* parameter object                       */
  UNUR_GEN   *comp[2]; /* array of generator objects (components)*/
  UNUR_GEN   *gen;     /* generator object for mixture           */

  double prob[2];      /* array of probabilities                 */

  int    i;            /* loop variable                          */
  double x;            /* will hold the random number            */

  /* Create generators for components                            */
  distr = unur_distr_normal(NULL,0);   /* Gaussian distribution  */
  unur_distr_cont_set_domain(distr,-UNUR_INFINITY,0);
  par = unur_pinv_new(distr);          /* choose method PINV     */
  comp[0] = unur_init(par);            /* initialize             */
  unur_distr_free(distr);              /* free distribution obj. */

  distr = unur_distr_exponential(NULL,0); /* Exponential distr.  */
  par = unur_pinv_new(distr);          /* choose method PINV     */
  comp[1] = unur_init(par);            /* initialize             */
  unur_distr_free(distr);              /* free distribution obj. */

  /* Probabilities for components (need not sum to 1)            */
  prob[0] = 0.4;    
  prob[1] = 0.3;

  /* Create mixture */
  par = unur_mixt_new(2,prob,comp);

  /* We want to use inversion for the mixture as well.           */
  /* (Thus the above order of the components is important!)      */
  unur_mixt_set_useinversion(par,TRUE);

  /* Initialize generator object                                 */
  gen = unur_init(par);

  /* we do not need the components any more */
  for (i=0; i<2; i++)
    unur_free(comp[i]);

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
