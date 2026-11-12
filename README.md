# Alzheimer's Disease Classification from PET Scans with GAN Data Augmentation

Master's thesis project (Software Engineering, Abdelhamid Mehri Constantine 2 University, 2024).

Medical imaging datasets are small and unbalanced, which limits how well deep learning models can detect Alzheimer's disease early. This project uses a **Generative Adversarial Network (GAN)** to generate synthetic PET scans, then trains a **convolutional neural network (CNN)** on the augmented data to classify patients.

## Data

- PET brain scans from the [ADNI](https://adni.loni.usc.edu/) database (DICOM format).
- Cognitive groups: Alzheimer's disease (AD), mild cognitive impairment (MCI) and cognitively normal (CN).
- The data is **not included** in this repository: ADNI access requires an approved application.

## Pipeline

| Step | Notebook | What it does |
|---|---|---|
| 1. Augmentation | [`Data augmentation.ipynb`](Data%20augmentation.ipynb) | Loads DICOM slices with `pydicom`, resizes them to 160×160, trains a GAN (dense generator and discriminator) and writes synthetic scans back as DICOM files. |
| 2. Classification | [`custom data generator finale.ipynb`](custom%20data%20generator%20finale.ipynb) | Custom Keras data generator (resize to 128×128 with `scikit-image`), train/validation split, CNN (3 Conv2D + MaxPooling blocks, Dense 512, Dropout), training and evaluation with scikit-learn metrics and confusion matrices. |

## Results

The final model reported in the thesis reached **98.39% accuracy** on the 3-class task (AD / MCI / CN), trained on the GAN-augmented dataset and compared with reference models.

The notebooks in this repository contain the augmentation pipeline and an earlier binary baseline classifier, so the accuracy printed in their outputs is lower than the final result.

## Tech stack

Python · TensorFlow / Keras · NumPy · pydicom · scikit-image · scikit-learn · Matplotlib · Jupyter

## Run it

```bash
pip install tensorflow pydicom scikit-image scikit-learn matplotlib numpy
jupyter notebook
```

Set the data paths at the top of each notebook to your local ADNI export, then run `Data augmentation.ipynb` before `custom data generator finale.ipynb`.

## Author

Ridha Lezzar · [LinkedIn](https://linkedin.com/in/ridha-lezzar-8322222b5)
