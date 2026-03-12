# Plutchik-48: A Geometrically Coherent RGB Manifold for Affective Feature Extraction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18831701.svg)](https://doi.org/10.5281/zenodo.18831701)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Notice:** This repository contains the official dataset, validation scripts, and an interactive web application directly associated with our manuscript submitted to **The Visual Computer**. 

## 📖 Overview

The computational analysis of emotion in visual art is fundamentally constrained by the "subjectivity gap"—the absence of a normalized, quantifiable ground truth connecting low-level colorimetric data to high-level emotional semantics. 

**Plutchik-48** provides a deterministic feature space that systematically encodes 48 distinct RGB anchor points within a hierarchical taxonomy grounded in Robert Plutchik's Psycho-Evolutionary Theory. By formalizing emotional complexity into a geometrically coherent paradigm (utilizing hue constancy and weighted vector integration), this repository offers a transparent, reproducible, and psychologically grounded alternative to "black-box" deep learning models.

---

## 📝 Citation Request

If you use this dataset, pipeline, or web application in your research, **we strongly urge you to cite our manuscript**. 

**Cite as:**
> Manimaran, A., & Lingaswamy, S. (2026). Plutchik-48: A Geometrically Coherent RGB Manifold for Affective Feature Extraction. *Submitted to The Visual Computer*.

**BibTeX:**
```bibtex
@article{manimaran2026plutchik48,
  author  = {Manimaran, Amudhan and Lingaswamy, Sindhia},
  title   = {Plutchik-48: A Geometrically Coherent RGB Manifold for Affective Feature Extraction},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript under review at The Visual Computer. Dataset DOI: https://doi.org/10.5281/zenodo.18831701}
}

```

---

## 📂 Repository Structure

```text
EMOTION-ART-DETECTION/
│
├── dataset/                     # The 100-artwork corpus used for the Ablation Study
│   ├── Cubism/
│   ├── Impressionism/
│   ├── Nihonga/
│   └── Romanticism/
│
├── data/                        
│   ├── images/                  # Raw/additional image data
│   └── labels.csv               # Corresponding emotional target labels
│
├── static/ & templates/         # Web Application UI assets
│   ├── static/styles.css        # CSS styling for the interface
│   └── templates/               # HTML templates (upload.html, result.html, hsi-reference.html)
│
├── plutchik48_dataset.csv       # The complete 48-point RGB manifold (emotion label, tier, RGB, hex)
├── app.py                       # Main Flask web application for interactive testing
├── validate_wea.py              # Script to compute Weighted Emotion Accuracy (WEA) for the dataset
├── qvr.py                       # Script to generate Qualitative Visual Results
│
├── ablation_table.csv           # Final extracted metrics for Configurations A, B, and C
├── statistical_table.csv        # T-test and Cohen's d statistical effect sizes
├── requirements.txt             # Python dependencies
└── README.md                    # This file

```

---

## ⚙️ Installation & Dependencies

To replicate our experiments, run the WEA validation, or launch the interactive web app, please ensure you have Python installed.

1. Clone this repository:

```bash
git clone https://github.com/AmudhanManimaran/Emotion-Detection-Through-Colors-Analysis.git
cd Emotion-Detection-Through-Colors-Analysis

```

2. Install the required dependencies:

```bash
pip install -r requirements.txt

```

*(Key dependencies include: `Flask`, `numpy`, `pandas`, `scikit-learn` (for GMM), `opencv-python`, `matplotlib`)*

---

## 🚀 Usage Guidelines

### 1. Interactive Web Application

We have provided a Flask-based web interface to allow users to interactively upload any digital artwork, extract its $K=25$ chromatic fingerprint, and instantly map it to the Plutchik-48 manifold.

```bash
python app.py

```

*Once running, navigate to `http://127.0.0.1:5000` in your web browser to access the upload portal and view semantic emotional results.*

### 2. Validating the Ablation Study (WEA)

To reproduce the strictly monotonic gains and calculate the Weighted Emotion Accuracy (WEA) across the four stylistic genres (Cubism, Impressionism, Nihonga, Romanticism) as detailed in our manuscript:

```bash
python validate_wea.py

```

*This script processes the images in the `dataset/` directory and outputs the WEA scores matching the manuscript's ablation tables.*

### 3. Generating Qualitative Visual Results

To generate the visual palette grids and probability distributions (as seen in Figure 7 of the manuscript):

```bash
python qvr.py

```

*This will output high-resolution figures (e.g., `Qualitative_Visual_Results_Final.pdf/png`) mapping extracted GMM centroids to their nearest emotional anchors.*

### 4. The Plutchik-48 Dataset (CSV)

The core dataset is available as `plutchik48_dataset.csv` and contains all 48 emotional centroids with their RGB coordinates, hexadecimal codes, derivation tier, and psychophysical derivation logic. It can be loaded directly into any ML framework:

```python
import pandas as pd
df = pd.read_csv('data/labels.csv')
print(df.head())

```

---

## 📜 License

This project is licensed under the MIT License. Academic and commercial use is permitted, provided appropriate credit is given via citation of the associated manuscript.

## ✉️ Contact

For any questions regarding the dataset, derivation rules, or methodology, please reach out to:

* **Amudhan Manimaran:** amudhanmanimaran.am@gmail.com
* **Dr. Sindhia Lingaswamy:** sindhia@nitt.edu

```

***

Your GitHub repository is now fully submission-ready. The clone command works perfectly, the dataset is highlighted as the star of the show, the figure reference is correct, and the citation block is rock solid. 

Shall we move on and draft that final, highly persuasive Cover Letter to the Editor?

```