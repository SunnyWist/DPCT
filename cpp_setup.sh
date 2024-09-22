cd ./DPCT/od_mstar3
python3 setup.py clean
python3 setup.py build_ext --inplace
cd ../astarlib3
python3 setup.py clean
python3 setup.py build_ext --inplace
cd ../..
