CC=g++
CXXFLAGS=-W -Wall -ansi -pedantic
CXXFLAGS += -std=c++17

sources = main.cpp Matrix.cpp mnist_reader.cpp 

EXEC=MNT

all: $(EXEC)

MNT: main.o Matrix.o mnist_reader.o
	$(CC) -o $@ $^ $(CXXFLAGS)

main.o: main.cpp Matrix.hpp mnist_reader.hpp
	$(CC) -c $^ $(CXXFLAGS)

Matrix.o: Matrix.cpp Matrix.hpp
	$(CC) -c $^ $(CXXFLAGS) 

mnist_reader.o: mnist_reader.cpp mnist_reader.hpp
	$(CC) -c $^ $(CXXFLAGS)

depend:
	makedepend $(sources)

