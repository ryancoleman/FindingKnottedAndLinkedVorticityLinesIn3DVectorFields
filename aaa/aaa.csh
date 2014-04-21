#set environmental variable $INPUTAAA equal to input file
export INPUTAAA=$1
export INPUTAAAOUT=$1.kappa.$2.out
./test ${INPUTAAA} ${INPUTAAAOUT} $2

