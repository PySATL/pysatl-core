/* ------------------------------------------------------------- */
/* File: example_discr.c                                         */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Example how to sample from a discrete univariate distribution.*/

/* ------------------------------------------------------------- */

int main(void)
{
  int    i;
  double param = 0.3;

  double probvec[10] = {1.0, 2.0, 3.0, 4.0, 5.0,\
                        6.0, 7.0, 8.0, 4.0, 3.0};

  /* Declare the three UNURAN objects.                           */
  UNUR_DISTR *distr1, *distr2;    /* distribution objects        */
  UNUR_PAR   *par1, *par2;        /* parameter objects           */
  UNUR_GEN   *gen1, *gen2;        /* generator objects           */

  /* First distribution: defined by PMF.                         */
  distr1 = unur_distr_geometric(&param, 1);
  unur_distr_discr_set_mode(distr1, 0);

  /* Choose a method: DARI.                                      */
  par1 = unur_dari_new(distr1);
  gen1 = unur_init(par1);

  /* It is important to check if the creation of the generator   */
  /* object was successful. Otherwise `gen' is the NULL pointer  */ 
  /* and would cause a segmentation fault if used for sampling.  */
  if (gen1 == NULL) {
     fprintf(stderr, "ERROR: cannot create generator object\n");
     exit (EXIT_FAILURE);
  }
  
  /* Second distribution: defined by (finite) PV.                */
  distr2 = unur_distr_discr_new();
  unur_distr_discr_set_pv(distr2, probvec, 10);

  /* Choose a method: DGT.                                       */
  par2 = unur_dgt_new(distr2);
  gen2 = unur_init(par2);
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
  unur_distr_free(distr1);
  unur_distr_free(distr2);
  unur_free(gen1);
  unur_free(gen2);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */
