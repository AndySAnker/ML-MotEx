[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6263f48f5b900913a0195c4f)  |  [Paper](https://www.nature.com/articles/s41524-022-00896-3)

**ML-MotEx_Step1_2**: Generates a catalogue of structure motifs (Step 1) from a starting model and fit these to a PDF (Step 2):

Option 1: Install using requirement.txt files
```bash
cd ML-MotEx_Step1+2
```
```bash
bash install.sh
```

Option 2: Run ML-MotEx using a singularity container in the folder "Singularity"

**ML-MotEx_Step3_4**: The results from the fits are handed to a ML algorithm (Step 3), and using SHAP (SHapley Additive exPlanation) values, the machine identifies which atoms are important for the fit quality (Step 4), such that structural motifs can be extracted from a dataset.

Option 1: Instal using requirement.txt files
```bash
cd ML-MotEx_Step3+4
```
```bash
bash install.sh
```

Option 2: Run ML-MotEx using a singularity container in the folder "Singularity"





