/*****************************************************************************
 *                                                                           *
 *          UNURAN -- Universal Non-Uniform Random number generator          *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   FILE: unur_methods_source.h                                             *
 *                                                                           *
 *   PURPOSE:                                                                *
 *         defines macros for checking method in generator objects.          *
 *         the bitmasks are defined in file unur_methods.h.                  *
 *                                                                           *
 *****************************************************************************
 *                                                                           *
 *   Copyright (c) 2000-2011 Wolfgang Hoermann and Josef Leydold             *
 *   Department of Statistics and Mathematics, WU Wien, Austria              *
 *                                                                           *
 *   This program is free software; you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation; either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program; if not, write to the                           *
 *   Free Software Foundation, Inc.,                                         *
 *   59 Temple Place, Suite 330, Boston, MA 02111-1307, USA                  *
 *                                                                           *
 *****************************************************************************/

/*---------------------------------------------------------------------------*/
#ifndef UNUR_METHODS_SOURCE_H_SEEN
#define UNUR_METHODS_SOURCE_H_SEEN
/*---------------------------------------------------------------------------*/

/*****************************************************************************/
/**  Bitmask to indicate methods                                            **/
/*****************************************************************************/

#include <methods/unur_methods.h>

/*****************************************************************************/
/**  Macros                                                                 **/
/*****************************************************************************/

/*---------------------------------------------------------------------------*/
/* check if parameter object is of correct type, return 0 otherwise       */

#define _unur_check_par_object( par,type ) \
  if ( (par)->method != UNUR_METH_##type ) { \
    _unur_error(#type,UNUR_ERR_PAR_INVALID,""); \
    return (UNUR_ERR_PAR_INVALID); } \
  COOKIE_CHECK((par),CK_##type##_PAR,UNUR_ERR_COOKIE)

/*---------------------------------------------------------------------------*/
/* check if generator object is of correct type, return 0 otherwise          */

#define _unur_check_gen_object( gen,type,rval ) \
  if ( (gen)->method != UNUR_METH_##type ) { \
    _unur_error((gen)->genid,UNUR_ERR_GEN_INVALID,""); \
    return rval; } \
  COOKIE_CHECK((gen),CK_##type##_GEN,UNUR_ERR_COOKIE)

/*---------------------------------------------------------------------------*/
#endif  /* UNUR_METHODS_SOURCE_H_SEEN */
/*---------------------------------------------------------------------------*/
