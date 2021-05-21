all: hog
hog: hog.cpp
	g++ -g -O3 hog.cpp -o hog `pkg-config --cflags --libs opencv`

.PHONY: clean
clean:
	rm hog
