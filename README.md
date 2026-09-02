# CardioSynth-LSTM

**Predictive Arrhythmia Forecasting and Synthetic ECG Synthesis using LSTM-Autoencoders and Time-Series GANs (TimeGAN)**

**Student:** Syed Umer  
**Roll No.:** 64536  
**Course:** ANN Lab  
**Department:** Artificial Intelligence, Iqra University

## Project Overview
CardioSynth-LSTM combines LSTM-Autoencoder based ECG reconstruction/anomaly detection, a recurrent TimeGAN-style Wasserstein GAN for synthetic ventricular ectopic ECG generation, and downstream sequence classification.

## Dataset
The project uses the MIT-BIH Arrhythmia Database through WFDB/PhysioNet. Raw MIT-BIH files are not included in this repository; the notebook downloads the required records.

## Main Files
- `notebook/Final_Syed_Umer_64536_ANN_Lab_project.ipynb` — complete notebook
- `CardioSynth_LSTM_Project.py` — Python source extracted from notebook code cells
- `outputs/figures/` — figures
- `outputs/models/` — trained weights
- `outputs/results/` — result files
- `data/README.md` — dataset note

## Implementation Note
The adversarial model is a **TimeGAN-style recurrent WGAN**, not the full canonical TimeGAN architecture. It uses recurrent LSTM Generator and Discriminator/Critic networks with Wasserstein loss, dropout, L2 weight decay, and critic weight clipping.
