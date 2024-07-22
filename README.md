# Human Activity Recognition with Consumer Devices and Real-Life Perspectives

[![GH Deployment](https://img.shields.io/badge/GitHub%20Pages-Deployment-green.svg)](https://matey97.github.io/thesis)

This repository collects all the resources employed during the development of the doctoral thesis in computer science titled *"Human Activity Recognition with Consumer Devices and Real-Life Perspectives"*, authored by 
Miguel Matey Sanz <a href="https://orcid.org/0000-0002-1189-5079"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>
and supervised by Carlos Granell Canut <a href="https://orcid.org/0000-0003-1004-9695"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>
and Sven Casteleyn <a href="https://orcid.org/0000-0003-0572-5716"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>
at the [Universitat Jaume I](www.uji.es).

The repository employs [Quarto](https://quarto.org) to present the main outcomes of the thesis in a book format deployed in [GitHub Pages](https://matey97.github.io/thesis). The generated Quarto Book consitutes a "lite" version of the thesis. Therefore, the thesis document should be consulted for more context, references, or discussions.

## Contents
The repository includes all the data, code and other resources employed throughout the develoment of the thesis:

- `data`: contains the data employed in each one of the chapters of the thesis.
- `figs`: contains the figures for each chapter of the thesis.
- `libs`: Python library contanining all the code employed to execute the experiments (`libs/chapter*/pipeline/`) and analyses (`libs/chapter*/analysis/`) presented in the thesis.
- `reference`: contains `.qmd` files with the documentation of the most important code resources in `libs`. Generated using [quartodoc](https://github.com/machow/quartodoc/) :rocket:.
- `*.qmd` files: Quarto Markdown documents
- `*.ipynb` files: Jupyter notebooks containing the analyses whose results are presented in the thesis.
- `requirements.txt`: Python libraries employed to execute experiments and analyses. All these experiments and analyses have been executed using Python 3.9.
- `.docker`: contains a Dockerfile to build a Docker image with a computational environment to reproduce the experiments and analyses.

## Reproducibility
From the begining of this thesis, the reproducibility of the results has been a paramounth objective. Therefore, all the outcomes presented in the thesis document can be reproduced using the `*.ipynb` notebooks. In addition, since all the scripts employed for the execution of experiments are provided, their replicability is also possible.

### Reproducibility setup
Several options to setup a computational environment to reproduce the analyses are offered: online and locally.

#### Reproduce online with Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/matey97/thesis/HEAD)

[Binder](https://mybinder.readthedocs.io/en/latest/) allows to create custom computing environments in the cloud so it can be shared to many remote users.
To open the Binder computing environment, click on the "Binder" badge above. You can also open the Binder enviroment while exploring the Quarto Book clicking in the "Launch Binder" link provided in some sections.

> [!NOTE]
> Building the computing enviroment in Binder can be slow.

#### Reproduce locally
Install Python 3.9, download or clone the repository, open a command line in the root of the directory and install the required software executing the following command:

```bash
pip install -r requirements.txt
```

> [!TIP]
> The usage of a virtual enviroment such as the ones provided by [Conda](https://conda.io/projects/conda/en/latest/index.html) or [venv](https://docs.python.org/3/library/venv.html) are recommended.

#### Reproduce locally with Docker
Install [Docker](https://www.docker.com) for building an image based on the provided `.docker/Dockerfile` with a Jupyter environment and running a container based on the image.

Download the repository, open a command line in the root of the directory and:

1. Build the image (don't forget the final `.`):

```bash
docker build --file .docker/Dockerfile --tag thesis .
```

2. Run the image:

```bash
docker run -it -p 8888:8888 thesis
```

3. Click on the login link (or copy and paste in the browser) shown in the console to access to a Jupyter environment.

### Reproduce the analyses
The Python scripts employed to execute the experiments described in the thesis are located in `libs/chapter*/pipeline/[n]_*.py`, where `n` determines the order in which the scripts must be executed. The reproduction of these scripts is not needed since their outputs are already stored in the `data/chapter*/` directories.

> [!NOTE]
> When executing a script with a component of randomness (i.e., ML and DL models), the obtained results might change compared with the reported ones.

> [!CAUTION]
> It is not recommended to execute these scripts, since they can run for hours, days or weeks depending on the computer's hardware.

To reproduce the outcomes presented in the thesis, open the desired Jupyter notebook (`*.ipynb`) file and execute its cells to generate reported results from the data generated in the experiments (`libs/chapter*/pipeline/[n]_*.py` scripts).

## License

[![License!: GPL v3](https://img.shields.io/badge/Code%20License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
[![License: ODbL](https://img.shields.io/badge/Data%20License-ODbL-brightgreen.svg)](https://opendatacommons.org/licenses/odbl/) 
[![License: CC BY-SA 4.0](https://img.shields.io/badge/Documents%20License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

All the code contained in the `.ipynb` notebooks and the `libs` folder are licensed under the [GPL-3.0 License](LICENSE).

The data contained in the `data` folder is licensed under the [Open Data Commons Open Database License](https://opendatacommons.org/licenses/odbl/1.0) (ODbL).

The remaining documents included in this repository are licensed under the [Creative Commons Attribution-ShareAlike](https://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0).

## Funding

This thesis has been funded by the Spanish Ministry of Universities with a predoctoral grant (FPU19/05352) and a research stay grant (EST23/00320). Financial support for derived activities of this dissertation (e.g., publications, conferences, etc.) was received from the SyMptOMS-ET project (PID2020-120250RB-I00), funded by MICIU/AEI/10.13039/501100011033.