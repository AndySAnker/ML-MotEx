# Machine Learning based Motif Extractor (ML-MotEx)

We provide our code for Machine Learning based Motif Extractor, which is a tool to extract motifs from numerous fits using interpretable machine learning.


One of the bottlenecks in structural analysis using e.g. Pair Distribution Function (PDF) analysis or other scattering methods is identifying an atomic model for structure refinement. Recently, new modelling approaches have made it possible to test thousands of models against a dataset in an automated manner, but one of the challenges when using such methods is analyzing the output, i.e. extracting structural information from the thousands of fits in a meaningful way. 
We here use interpretable machine learning to identify structural motifs present in nanomaterials from PDFs based on an automated modelling approach. Our Machine Learning based Motif Extractor (ML-MotEx) first builds a catalogue of hundreds or thousands of candidate structure motifs which are all ‘cutouts’ from a chosen starting structure (Step 1), and then fit these individual models to a dataset (Step 2). The results from these fits are then handed to a Machine Learning algorithm (Step 3), and using SHAP (SHapley Additive exPlanation) values, the machine identifies which atoms are important for the fit quality (Step 4), such that structural motifs can be extracted from a PDF.

![alt text](ML_MotEx_Overview.png "Title")

We have demonstrated the use of the algorithm on data from 4 different chemical systems consisting of disordered materials and ionic clusters. Furthermore, we show that the algorithm achieves comparable results using 4 different starting models but the same dataset. The method presented here opens for a new type of modelling where each atom or structural feature in a model is assigned an importance value for the fit quality based on Machine Learning. 
The results are published:
