.PHONY: all aaa

all: aaa

download:
	curl https://www.automl.org/wp-content/uploads/2019/10/cplex_regions200.zip -O
	unzip cplex_regions200.zip 

aaa: 
	cd GenericWrapper4AC/runsolver/runsolver-3.4.0/src && $(MAKE)
	mkdir -p GenericWrapper4AC/genericWrapper4AC/binaries
	cp GenericWrapper4AC/runsolver/runsolver-3.4.0/src/runsolver GenericWrapper4AC/genericWrapper4AC/binaries/runsolver

clean:
	cd GenericWrapper4AC/runsolver/runsolver-3.4.0/src && $(MAKE) clean
	rm -rf GenericWrapper4AC/genericWrapper4AC/binaries
	
install:
	pip3 install -r requirements.txt