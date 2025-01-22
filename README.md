# **SAMson: An Automated Brain Extraction Tool for Rodents Using SAM**  

Daniel Panadero Soler<sup>1</sup>, Mohamed Kotb Selim<sup>1</sup>, Patricia Martínez-Tazo<sup>1</sup>, Emma Muñoz-Moreno<sup>2</sup>, Pedro Ramos-Cabrer<sup>3,4</sup>, Pilar López-Larrubia<sup>5</sup>, Silvia De Santis<sup>1</sup>, Santiago Canals<sup>1</sup>, and Antonio Pertusa<sup>6</sup>  

<sup>1</sup>Instituto de Neurociencias, CSIC-UMH, Sant Joan d’Alacant, Spain  
<sup>2</sup>IDIBAPS, Barcelona, Spain  
<sup>3</sup>CIC biomaGUNE, BRTA, Donostia–San Sebastián, Spain  
<sup>4</sup>Ikerbasque, Basque Foundation for Science, Bilbao, Spain  
<sup>5</sup>Instituto de Investigaciones Biomédicas Sols-Morreale, CSIC-UAM, Madrid, Spain  
<sup>6</sup>University Institute for Computer Research, University of Alicante, Alicante, Spain  

[[`Paper`](https://www.biorxiv.org/content/10.1101/2024.03.07.583982)] [[`Data`](https://doi.org/10.20350/digitalCSIC/17000)] [[`Code`](https://github.com/CanalsLab/SAMson/blob/main/SAMson.ipynb)][[`BibTeX`](#citation)]

## Abstract 
Accurate brain extraction is a critical step in the analysis of rodent head magnetic resonance imaging (MRI) data. However, current methods often encounter difficulties in handling the diverse range of imaging setups, resolutions, and experimental conditions commonly found in this field. **SAMson (*SAM for Segmentation Of Neuroimages*)** is an automated tool based on the Segment Anything Model (SAM), designed for robust rodent brain extraction. It integrates a bounding box generator and a mask prediction pipeline, offering fully automated and semi-automated modes to address varying experimental complexities. SAMson has been evaluated using three multi-centre rodent MRI datasets annotated at the pixel level, differing in acquisition parameters, resolution, and animal age groups. The tool demonstrates superior performance to existing methods (BET, RBM, BEN) in terms of segmentation accuracy, with Jaccard indices exceeding 90% across datasets. Its semi-automated mode is particularly effective in challenging scenarios such as low-resolution images and cases requiring refined mask precision. By identifying errors at the individual slice level, SAMson enables rapid and targeted corrections, advancing translational neuroscience by supporting large-scale research workflows.  

## Installation 
To set up SAMson, execute the following commands:  
```
conda create -n samson python=3.10       # Create a new Conda environment  
conda activate samson                    # Activate the environment  
pip install -r path/to/requirements.txt  # Install the required dependencies  
jupyter-notebook                         # Launch Jupyter Notebook  
```

## Getting Started
To start using SAMson, proceed with the following steps:
1. Download the `requirements.txt` and `SAMson.ipynb` files from the repository.
2. Complete the installation steps outlined above. Additionally, download the following resources:
 - [SAM's ViT-H model checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).
 - [Mouse-X template](https://doi.org/10.20350/digitalCSIC/17000).
3. Open the Jupyter Notebook ([`SAMson.ipynb`](https://github.com/CanalsLab/SAMson/blob/main/SAMson.ipynb)) and follow the instructions to initialize and use the tool.

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











