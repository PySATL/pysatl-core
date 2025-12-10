#!/bin/bash

# check arguments 
test -z $1 && echo "Argument missing" && exit 1

test -f suppressions-valgrind && SUPPRESS="--suppressions=suppressions-valgrind"

VALGRINDOPTIONS="-v --tool=memcheck --leak-check=full --leak-resolution=high --num-callers=40 --show-reachable=yes --track-fds=yes $SUPPRESS"

##VALGRINDOPTIONS="-v --stats=yes --tool=memcheck --leak-check=full --leak-resolution=high --num-callers=40 --show-reachable=yes --track-fds=yes $SUPPRESS"

PROG=`echo $1 | sed -e "s#\./##"`

LOGFILENAME="valgrind-${PROG}"

echo "run valgrind on ${PROG} ..."

valgrind --log-file=${LOGFILENAME} ${VALGRINDOPTIONS} ./${PROG}

echo ""; echo "Summary:"; echo ""
grep ERROR ${LOGFILENAME}*
grep lost ${LOGFILENAME}*
grep "All heap blocks were freed" ${LOGFILENAME}*
echo ""; echo "========================="; echo ""

exit 0
