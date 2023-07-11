FROM andriusbern/vina-gpu:latest

COPY . "/diffdock"

WORKDIR "/diffdock"

RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117 \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html \
    install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas \
    "fair-esm[esmfold]"

CMD ["sleep", "infinity"]