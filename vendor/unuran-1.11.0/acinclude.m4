dnl
dnl AX_IEEE_COMP
dnl Check IEEE comparisons, whether "x != x" is true for NaNs
dnl -------------------------------------------------------------------------
AC_DEFUN([AX_IEEE_COMP],
[
AC_CACHE_CHECK([for IEEE comparisons], ac_cv_c_ieee_comparisons,
[AC_RUN_IFELSE([AC_LANG_SOURCE([[
#include <math.h>
int main (void) 
{ 
   int status; double inf, nan;
   inf = exp(1.0e10);
   nan = inf / inf;
   /* nan = 0.0 / 0.0; */
   status = (nan == nan);
   return status;
}]])],
[ac_cv_c_ieee_comparisons="yes"],
[ac_cv_c_ieee_comparisons="no"],
[ac_cv_c_ieee_comparisons="yes"])
])
if test "$ac_cv_c_ieee_comparisons" != no ; then
  AC_DEFINE([HAVE_IEEE_COMPARISONS], [1], 
      [Define to 1 if "x != x" is true for NaNs])
  AC_SUBST(HAVE_IEEE_COMPARISONS)
fi
]) # end of AX_IEEE_COMP


dnl
dnl AX_DIVIDE_BY_ZERO
dnl Check whether 1./0. works and results in "infinity"
dnl -------------------------------------------------------------------------
AC_DEFUN([AX_DIVIDE_BY_ZERO],
[
AC_CACHE_CHECK([for divide-by-zero], ac_cv_c_divide_by_zero,
[AC_RUN_IFELSE([AC_LANG_SOURCE([[
#include <math.h>
#include <float.h>
int main (void) 
{
   double x = 0.0;
   double inf = 1.0 / x;
   int status = (inf < DBL_MAX/2.);
   return status;
}]])],
[ac_cv_c_divide_by_zero="yes"],
[ac_cv_c_divide_by_zero="no"],
[ac_cv_c_divide_by_zero="yes"])
])
if test "$ac_cv_c_divide_by_zero" != no ; then
  AC_DEFINE([HAVE_DIVIDE_BY_ZERO], [1], 
      [Define to 1 if "1.0/0.0" results in INFINITY])
  AC_SUBST(HAVE_DIVIDE_BY_ZERO)
fi
]) # end of AX_DIVIDE_BY_ZERO


dnl
dnl AX_ADD_URNG_LIB([TAG],[UTAG],[NAME],[LIB],[PROG],[HEADER])
dnl Add support for library providing uniform random number generators
dnl via configure flag and check existance of library and header file
dnl -------------------------------------------------------------------------
AC_DEFUN([AX_ADD_URNG_LIB],
[
AC_ARG_WITH(urng_$1,
	[AS_HELP_STRING([--with-urng-$1],
	   [URNG: use $3 @<:@default=no@:>@])],
	[case "${withval}" in
	   yes) ;; no) ;; 
	   *) AC_MSG_ERROR(bad value '${withval}' for --with-urng-$1) ;;
	esac],
	[with_urng_$1=no])
AS_IF([test "x$with_urng_$1" != xno],
	[AC_CHECK_LIB([$4], [$5], [], [])
	 AC_CHECK_HEADERS([$6],
	   [AC_SUBST(UNURAN_SUPPORTS_$2,-DUNURAN_SUPPORTS_$2)
	    AC_DEFINE_UNQUOTED([UNURAN_HAS_$2], [1], [Define to 1 if you use $3.])],
	   [AC_MSG_FAILURE(
	      [PRINT_WITH_FRAME([--with-urng-$1 given, but '$6' not found])
	   ]) ]) ], [])
])


dnl
dnl AX_ADD_RMATH()
dnl Add support for Rmath standalone library from R project
dnl via configure flag and check existance of library 
dnl (Remark: AC_CHECK_HEADERS cannot be used as is. I have skipped the
dnl test for header file Rmath.h.)
dnl -------------------------------------------------------------------------
AC_DEFUN([AX_ADD_RMATH],
[
AC_ARG_WITH(Rmath,
	[AS_HELP_STRING([--with-Rmath],
	   [use Rmath library from R project @<:@EXPERIMENTAL@:>@ @<:@default=no@:>@])],
	[case "${withval}" in
	   yes) ;; no) ;; 
	   *) AC_MSG_ERROR(bad value '${withval}' for --with-Rmath) ;;
	esac],
	[with_Rmath=no])
AS_IF([test "x$with_Rmath" != xno],
	[AC_CHECK_LIB([Rmath], [bessel_k],[],
	   [AC_MSG_FAILURE(
	      [PRINT_WITH_FRAME([--with-Rmath given, but 'libRmath' not found])
	   ]) ]) ], [])
])


dnl
dnl PRINT_WITH_FRAME([STRING)]
dnl Print 'string' inside framed box
dnl -------------------------------------------------------------------------
AC_DEFUN([PRINT_WITH_FRAME],[
`echo "$1" | \
  $SED 's/\(.*\)/*   \1   */;h;s/./*/g;p;s/./ /g;s/./*/;s/.$/*/;p;x;p;x;p;x;s/./*/g'`
]) # end of PRINT_WITH_FRAME
