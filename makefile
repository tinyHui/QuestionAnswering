clean:
	rm -f utils.*.so
	rm -f utils.c
	rm -rf build
all:
	python setup.py build_ext --inplace
