/* ------------------------------------------------------------- */
/* File: example_FuncStr.c                                       */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Example how to define the PDF for a continuous univariate     */
/* distribution using a function string.                         */

/* ------------------------------------------------------------- */

int main(void)
{
  UNUR_DISTR *distr;    /* distribution object */
  char *pdfstr;

  /* Get empty distribution object for a continuous distribution */
  distr = unur_distr_cont_new();

  /* Set PDF using function string */
  unur_distr_cont_set_pdfstr(distr,"1-x*x");
  unur_distr_cont_set_domain(distr,-1.,1.);
 
  /* Read function string from distribution object */
  pdfstr = unur_distr_cont_get_pdfstr(distr);
  printf("functionstring: %s\n",pdfstr);

  /* Destroy distribution object and clear memory */
  unur_distr_free(distr);
  free (pdfstr);

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */
