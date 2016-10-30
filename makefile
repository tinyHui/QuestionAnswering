clean:
	rm -f preprocess/utils.*.so
	rm -rf build/
	rm -f preprocess/utils.c
all:
	env/bin/python setup.py build_ext --inplace
