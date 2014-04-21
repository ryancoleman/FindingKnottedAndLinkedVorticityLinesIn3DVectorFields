#set environmental variable $INPUTAAA equal to input file
export INPUTAAA=$1
export INPUTAAAOUT=$1.kappa.$2.num.multi.$3.out
./testmnr ${INPUTAAA} ${INPUTAAAOUT} $2 $3

