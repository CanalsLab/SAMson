# **SAMson: An Automated Brain Extraction Tool for Rodents Using SAM**

Daniel Panadero Soler<sup>1</sup>, Mohamed Kotb Selim<sup>1</sup>, Patricia Martínez-Tazo<sup>1</sup>, Emma Muñoz-Moreno<sup>2</sup>, Pedro Ramos-Cabrer<sup>3,4</sup>, Pilar López-Larrubia<sup>5</sup>, Silvia De Santis<sup>1</sup>, Santiago Canals<sup>1</sup>, and Antonio Pertusa<sup>6</sup>

<sup>1</sup>Instituto de Neurociencias, CSIC-UMH, Sant Joan d’Alacant, Spain
<sup>2</sup>IDIBAPS, Barcelona, Spain
<sup>3</sup>CIC biomaGUNE, BRTA, Donostia–San Sebastián, Spain
<sup>4</sup>Ikerbasque, Basque Foundation for Science, Bilbao, Spain
<sup>5</sup>Instituto de Investigaciones Biomédicas Sols-Morreale, CSIC-UAM, Madrid, Spain
<sup>6</sup>University Institute for Computer Research, University of Alicante, Alicante, Spain

[[`Paper`](https://www.biorxiv.org/content/10.1101/2024.03.07.583982)] [[`Data`](https://doi.org/10.20350/digitalCSIC/17000)] [[`Code`](https://github.com/CanalsLab/SAMson/blob/main/notebooks/SAMson.ipynb)] [[`Package`](https://github.com/CanalsLab/SAMson/tree/main/package)] [[`BibTeX`](#citation)]

## Abstract
Accurate brain extraction is a critical step in the analysis of rodent head magnetic resonance imaging (MRI) data. However, current methods often encounter difficulties in handling the diverse range of imaging setups, resolutions, and experimental conditions commonly found in this field. **SAMson (*SAM for Segmentation Of Neuroimages*)** is an automated tool based on the Segment Anything Model (SAM), designed for robust rodent brain extraction. It integrates a bounding box generator and a mask prediction pipeline, offering fully automated and semi-automated modes to address varying experimental complexities. SAMson has been evaluated using seven multi-centre rodent MRI datasets annotated at the pixel level, differing in acquisition parameters, contrast (T2-weighted, T2\*-weighted and diffusion-weighted images), resolution, anteroposterior coverage and animal age groups. The tool outperforms the nine methods tested (BET, BET4Animal, RBM, BEN, SHERM, ANTs, deepbet, MedSAM and SAM2) in segmentation accuracy, significantly in 87 of 90 dataset–method comparisons, with mean Jaccard indices of 89.0–94.7%. Its semi-automated mode is particularly effective in challenging scenarios such as low-resolution images and cases requiring refined mask precision. Because its errors are confined to individual slices, SAMson enables rapid and targeted corrections, advancing translational neuroscience by supporting large-scale research workflows.

## Installation
To set up SAMson, execute the following commands in the repository folder:
```
conda create -n samson python=3.10 # Create a new Conda environment
conda activate samson # Activate the environment
pip install "package/dist/samson_mri-1.0.0-py3-none-any.whl[app]" # Install SAMson and the required dependencies
samson download-model # Download SAM's ViT-H model checkpoint (2.4 GB)
samson example # Check the installation on the bundled example
```
Tested versions: `package/source/environment.yml`. Details: [`package/guides/01_installation.md`](package/guides/01_installation.md).

## Getting Started
To start using SAMson, proceed with the following steps:
1. Complete the installation steps outlined above. The Mouse-X template is included in the package.
2. Segment your NIfTI images with the graphical application (`samson app`; [video tutorial](package/app_tutorial/)), the command line (`samson segment <folder> -o <output folder> --template <first> <last>`) or the Python API. See the [guides](package/guides/).
3. To use the Jupyter Notebook ([`notebooks/SAMson.ipynb`](notebooks/SAMson.ipynb)) or to reproduce the article's results, download the [data](https://doi.org/10.20350/digitalCSIC/17000) and place its `datasets/` and `outputs/` folders in the repository folder (see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

## Citation
If you use SAMson in your research, please use the following BibTeX entry.
```
@article {Panadero Soler2024.03.07.583982,
author = {Panadero Soler, Daniel and Kotb Selim, Mohamed and Mart{\'\i}nez-Tazo, Patricia and Mu{\~n}oz-Moreno, Emma and Ramos-Cabrer, Pedro and L{\'o}pez-Larrubia, Pilar and De Santis, Silvia and Canals, Santiago and Pertusa, Antonio},
title = {SAMson: an automated brain extraction tool for rodents using SAM},
elocation-id = {2024.03.07.583982},
year = {2025},
doi = {10.1101/2024.03.07.583982},
publisher = {Cold Spring Harbor Laboratory},
URL = {https://www.biorxiv.org/content/early/2025/01/15/2024.03.07.583982},
eprint = {https://www.biorxiv.org/content/early/2025/01/15/2024.03.07.583982.full.pdf},
journal = {bioRxiv}
}

```
