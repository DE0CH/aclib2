.PHONY: all build install clean

all: build

download:
	curl https://www.automl.org/wp-content/uploads/2019/10/cplex_regions200.zip -O
	unzip cplex_regions200.zip 

build: 
	cd GenericWrapper4AC/runsolver/runsolver-3.4.0/src && $(MAKE)
	mkdir -p GenericWrapper4AC/genericWrapper4AC/binaries
	cp GenericWrapper4AC/runsolver/runsolver-3.4.0/src/runsolver GenericWrapper4AC/genericWrapper4AC/binaries/runsolver
	cd random_forest_run && mkdir -p build && cd build && cmake .. && $(MAKE)

clean:
	cd GenericWrapper4AC/runsolver/runsolver-3.4.0/src && $(MAKE) clean
	rm -rf GenericWrapper4AC/genericWrapper4AC/binaries
	rm -rf random_forest_run/build
	
install:
	cd random_forest_run/build/python_package && pip3 install .
	pip3 install -r requirements.txt