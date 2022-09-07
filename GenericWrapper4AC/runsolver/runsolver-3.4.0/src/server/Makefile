.PHONY: clean run

all: server

server: server.cc
	$(CXX) -std=c++11 -o server server.cc

run: server
	./server

clean:
	rm -f server
