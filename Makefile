all: hog
hog: hog.cpp
	g++ hog.cpp -o hog `pkg-config --cflags --libs opencv`

.PHONY: clean
clean:
	rm hog
