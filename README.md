[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6263f48f5b900913a0195c4f)  |  [Paper] XXX

# Machine Learning based Motif Extractor (ML-MotEx)

We provide our code for Machine Learning based Motif Extractor (ML-MotEx), which is a tool to extract structural motifs from numerous fits using explainable machine learning.
ML-MotEx first builds a catalogue of hundreds or thousands of candidate structure motifs which are all ‘cutouts’ from a chosen starting structure (Step 1), and then fit these individual models to a dataset (Step 2). The results from these fits are then handed to a ML algorithm (Step 3), and using SHAP (SHapley Additive exPlanation) values, the machine identifies which atoms are important for the fit quality (Step 4), such that structural motifs can be extracted from a dataset.

Note that the code to step 2 presented here is specific for data analysis of Pair Distribution Function data. If data from other techniques is used, one can go directly to step 3+4 of the algorithm which will guide the user to set up the data (Step 1 + fits) in the appropriate manner and use step 3+4 of ML-MotEx.

![alt text](Images/ML_MotEx_Overview.png "Title")

One of the bottlenecks in structural analysis using e.g. Pair Distribution Function (PDF) analysis or other scattering methods is identifying an atomic model for structure refinement. Recently, new modelling approaches have made it possible to test thousands of models against a dataset in an automated manner, but one of the challenges when using such methods is analyzing the output, i.e. extracting structural information from the thousands of fits in a meaningful way. We here use explainable machine learning to identify structural motifs present in nanomaterials from PDFs based on an automated modelling approach.
We have demonstrated the use of the algorithm on data from 4 different chemical systems consisting of disordered materials and ionic clusters. Furthermore, we showed that the algorithm achieves comparable results using 4 different starting models but the same dataset. ML-MotEx opens for a new type of modelling where each atom or structural feature in a model is assigned an importance value for the fit quality based on Machine Learning. 

# How to use ML-MotEx
Follow these step if you want to use ML-MotEx locally on your own computer.

## Install requirements
See the [install](/Install) folder.

## Using step 1 and 2 of ML-MotEx
``` 
jupyter notebook ML-MotEx-Step1+2.ipynb
```

## Using step 3 and 4 of ML-MotEx
``` 
jupyter notebook ML-MotEx-Step3+4.ipynb
```
Or use ML-MotEx step 3+4 straightforwardly without any installion or downloads to your computer. Follow the instructions in our [Colab notebook](https://colab.research.google.com/github/AndySAnker/ML-MotEx/blob/main/ML_MotEx_Colab.ipynb) and try to play around.

### Citation
If you use our code or our results, please consider citing our paper. Thanks in advance!

```
@article{anker2022ML-MotEx,
  title={Extracting Structural Motifs from Pair Distribution Function Data of Nanostructures using Explainable Machine Learning},
  author={Andy S. Anker, Emil T. S. Kjær, Mikkel Juelsholt, Troels Lindahl Christiansen, Susanne Linn Skjærvø, Mads Ry Vogel Jørgensen, Innokenty Kantor, Daniel Risskov Sørensen, Simon J. L. Billinge, Raghavendra Selvan and Kirsten M. Ø. Jensen},
  booktitle={ChemRxiv},
  year={2022}}
```

### Contact
andy@chem.ku.dk

### Acknowledgments
Our code is developed based on the the following publications:
```
@article{LindahlChristiansen:kc5101,
title = "{Structure analysis of supported disordered molybdenum oxides using pair distribution function analysis and automated cluster modelling}",
author = "Lindahl Christiansen, Troels and Kjær, Emil T. S. and Kovyakh, Anton and Röderen, Morten L. and Høj, Martin and Vosch, Tom and Jensen, Kirsten M. Ø.",
journal = "Journal of Applied Crystallography",},}

@article{anker2021structural,
title={Structural Changes during the Growth of Atomically Precise Metal Oxido Nanoclusters from Combined Pair Distribution Function and Small-Angle X-ray Scattering Analysis},
author={Anker, Andy S and Christiansen, Troels Lindahl and Weber, Marcus and Schmiele, Martin and Brok, Erik and Kjær, Emil TS and Juhás, Pavol and Thomas, Rico and Mehring, Michael and Jensen, Kirsten M Ø},
journal={Angewandte Chemie},}
```

### LICENSE
This project is licensed under the Apache License Version 2.0, January 2004 - see the LICENSE file for details.
