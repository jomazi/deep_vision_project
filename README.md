# The Logo Generator

> Deep Vision SS 2019 \
> by L. Blessing and J. Ziegler 

A deep learning capable logo generator.

### Setup

- Clone repository into preferred directory

    ```
    git clone https://github.com/jomazi/deep_vision_project.git
    ```

- Create virtual environment

    ```
    cd deep_vision_project/
    virtualenv venv
    source venv/bin/activate
    ```

- Install required packages

    ```
    pip install -r requirements.txt
    ```

- Download data

    ```
    python ./src/download_data.py 
    ```
    
- Get ResNet and VAE encoded data

    ```
    python ./src/fcts/resnet_features.py
    python ./src/fcts/vae_features.py
    ```

- JupyterLab

    - Make Jupyter kernel for this project

        ```
        python -m ipykernel install --name deep_vision_project
        ```
        The kernel can later be selected in the JupyterLab to run the notebooks.

    - Add extensions

        ```
        jupyter nbextensions_configurator enable
        jupyter labextension install @jupyter-widgets/jupyterlab-manager
        ```
        Note: [Node.js](https://nodejs.org/en/) (version 10.16.0 LTS is used here) has to be installed for that!

    - Finally start JupyterLab

        ```
        jupyter lab
        ```

### Data

[LLD - Large Logo Dataset](https://data.vision.ee.ethz.ch/sagea/lld/#paper) published by
Alexander Sage, Eirikur Agustsson, Radu Timofte and Luc Van Gool