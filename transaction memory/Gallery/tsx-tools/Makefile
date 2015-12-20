all: rtm

rtm: rtm.o 
	g++ -mrtm -std=c++11 -pthread $< -o $@

rtm.o: rtm.cpp
	g++ -c -mrtm -std=c++11 -pthread $<

clean:
	rm -rf *o rtm