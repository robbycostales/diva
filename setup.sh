conda create -y --prefix ./env python=3.11
conda activate ./env
conda install -y mpi4py  # to avoid pip error
pip install -e .

# Fix error: https://stackoverflow.com/questions/78671850/importerror-cannot-import-name-runtime-version-from-google-protobuf
pip install --upgrade protobuf  # ==5.28.2