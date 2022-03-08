conda config --add channels diffpy
conda install --file "requirements_conda.txt"
pip3 install -r "requirements_pip.txt" || pip install -r "requirements_pip.txt"
