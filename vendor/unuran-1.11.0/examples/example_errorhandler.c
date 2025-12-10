/* ------------------------------------------------------------- */
/* File: example_errorhandler.c                                  */
/* ------------------------------------------------------------- */

/* Include UNURAN header file.                                   */
#include <unuran.h>

/* ------------------------------------------------------------- */

/* Simple example of an error handler that simply prints a       */
/* message to stderr.                                            */

/* ------------------------------------------------------------- */

void my_error_handler( 
        const char *objid,     /* id/type of object              */
	const char *file,      /* source file name (__FILE__)    */
	int line,              /* source line number (__LINE__)  */ 
	const char *errortype, /* "warning" or "error"           */
	int errorcode,         /* UNU.RAN error code             */
	const char *reason     /* short description of reason    */
     )   
{
  FILE *LOG = stderr;
  static int n = 0;

  fprintf(LOG,"\n");
  fprintf(LOG,"[[ %d ]] my_error_handler: [ %s ]\n",++n,errortype);
  fprintf(LOG,"\tobject = %s\n",objid);
  fprintf(LOG,"\tfile   = %s\n",file);
  fprintf(LOG,"\tline   = %d\n",line);
  fprintf(LOG,"\tcode   = [%#x] %s\n",errorcode,
	                             unur_get_strerror(errorcode));
  fprintf(LOG,"\treason = %s\n",reason);
  fprintf(LOG,"\n");

} /* end of my_error_handler() */

/* ------------------------------------------------------------- */

int main(void)
{
  /* Declare UNURAN object.                                      */
  UNUR_GEN   *gen;      /* generator object                      */

  /* Declare a point to hold old error handler.                  */
  UNUR_ERROR_HANDLER *default_error_handler = NULL;

  /* Set new error handler.                                      */
  default_error_handler = unur_set_error_handler( my_error_handler );

  /* The following statement causes an error */
  gen = unur_str2gen("normal(0.,-1)");

  exit (EXIT_SUCCESS);

} /* end of main() */

/* ------------------------------------------------------------- */

