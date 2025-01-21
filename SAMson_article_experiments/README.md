# SAMson Article Experiments

This folder contains a series of four Jupyter Notebooks comprising the complete set of experiments and analyses that underpin the evaluations presented in the [SAMson article](https://www.biorxiv.org/content/10.1101/2024.03.07.583982v2). Due to GitHub's size constraints, files in this directory may not render in the browser. Please download the files to view them locally.

## Contents
- **SAMson_experiments_1.ipynb**: accuracy evaluation of:
  1. The Full-Auto mode of SAMson.  
  2. The Semi-Auto mode of SAMson.  
  3. The object detection component of SAMson.  
  4. The oracle performance of SAMson.  
- **SAMson_experiments_2.ipynb**: accuracy evaluation of:
  1. SAMson using the [SAM2 model](https://github.com/facebookresearch/sam2).
  2. SAMson using the fine-tuned [MedSAM model](https://github.com/bowang-lab/MedSAM) (Jun Ma, Sumin Kim, Feifei Li, Mohammed Baharoon, Reza Asakereh, Hongwei Lyu, Bo Wang. Segment Anything in Medical Images and Videos: Benchmark and Deployment).
  3. Masks extracted using [BET](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/bet) (Smith SM. Fast robust automated brain extraction. Hum Brain Mapp. 2002 Nov;17(3):143-55. doi: 10.1002/hbm.10062. PMID: 12391568; PMCID: PMC6871816).
  4. Masks extracted using [SHERM](https://github.com/liu-yikang/SHERM-rodentSkullStrip) (Liu Y, Unsal HS, Tao Y, Zhang N. Automatic Brain Extraction for Rodent MRI Images. Neuroinformatics. 2020 Jun;18(3):395-406. doi: 10.1007/s12021-020-09453-z. PMID: 31989442; PMCID: PMC7343626).
- **SAMson_experiments_3.ipynb**: accuracy evaluation of masks extracted using:
  1. [Rodent U-net (RBM)](https://github.com/CAMRIatUNC/RodentMRISkullStripping) (Hsu LM, Wang S, Ranadive P, Ban W, Chao THH, Song S, et al. Automatic Skull Stripping of Rat and Mouse Brain MRI Data Using U-Net. Front Neurosci. 2020 Oct;14:568614. DOI: 10.3389/fnins.2020.568614).  
  2. [BEN](https://github.com/yu02019/BEN) (Yu Z, Han X, Xu W, Zhang J, Marr C, Shen D, Peng T, Zhang XY, Feng J. A generalizable brain extraction net (BEN) for multimodal MRI data from rodents, nonhuman primates, and humans. Elife. 2022 Dec 22;11:e81217. doi: 10.7554/eLife.81217. PMID: 36546674; PMCID: PMC9937657.).   
  3. [deepbet](https://github.com/wwu-mmll/deepbet) (Lukas Fisch, Stefan Zumdick, Carlotta Barkhau, Daniel Emden, Jan Ernsting, Ramona Leenings, Kelvin Sarink, Nils R. Winter, Benjamin Risse, Udo Dannlowski, Tim Hahn. "deepbet: Fast brain extraction of T1-weighted MRI using Convolutional Neural Networks". Computers in Biology and Medicine 2024; 179:108845).  
  4. [ANTs](https://github.com/ANTsX/ANTs) (Tustison, N, Cook, P, Holbrook, A, Johnson, H, Muschelli, J, Devenyi, G, Duda, J, Das, S, Cullen, N, Gillen, D, Yassa, M, Stone, J, Gee, J, Avants, B. "The ANTsX ecosystem for quantitative biological and medical imaging". Scientific Reports 2021; 11(1):9068).  
- **SAMson_experiments_4.ipynb**: all code necessary to reproduce:
  1. All figures included in the SAMson manuscript.  
  2. All statistical analyses presented in the SAMson article.
