clean:
    rm -f utils.*.so
    rm -f utils.c
    rm -rf build
all:
	swig -python -c -o preprocess/utils.c
	python setup.py build_ext --inplace