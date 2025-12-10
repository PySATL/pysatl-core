/* _unur_hypot
 *
 * Computs the value of sqrt(x^2 + y^2) in a way that avoids overflow.
 *
 * Replacement for missing C99 function hypot()
 *
 * Copied and renamed from gsl_hypot into _unur_hypot
 * by Josef Leydold, Tue Nov  1 11:17:37 CET 2011
 */

/* sys/hypot.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007 Gerard Jungman, Brian Gough, Patrick Alken
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <unur_source.h>

#if !HAVE_DECL_HYPOT
double _unur_hypot (const double x, const double y)
{
  double xabs = fabs(x) ;
  double yabs = fabs(y) ;
  double min, max;

  if (xabs < yabs) {
    min = xabs ;
    max = yabs ;
  } else {
    min = yabs ;
    max = xabs ;
  }

  if (min == 0) 
    {
      return max ;
    }

  {
    double u = min / max ;
    return max * sqrt (1 + u * u) ;
  }
} /* end of _unur_hypot() */
#endif
