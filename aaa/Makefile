all: 	test testmulti testmnr

test:	AAA.o
	gcc -O3 -Wno-deprecated -o test AAA.o -lm -lstdc++

testmulti:	AAA-multi.o
	gcc -O3 -Wno-deprecated -o testmulti AAA-multi.o -lm -lstdc++

testmnr:	AAA-mnr.o
	gcc -O3 -Wno-deprecated -o testmnr AAA-mnr.o -lm -lstdc++

%.o : %.cpp
	gcc -O3 -Wno-deprecated -c $< 

mettest:	MetricTest.cpp
		gcc -O3 -o mettest MetricTest.cpp -lm

clean:	
	rm *.o test testmulti mettest testmnr
