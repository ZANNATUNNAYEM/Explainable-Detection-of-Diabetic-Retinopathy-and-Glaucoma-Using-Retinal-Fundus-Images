# Comprehensive Progress Report: Work Done So Far & Future Roadmap

**Thesis Title:** Explainable Detection of Diabetic Retinopathy and Glaucoma Using Retinal Fundus Images  
**Department:** Department of Computer Science and Engineering, Brac University  
**Date:** October 2026  
**Supervisor:** Annajiat Alim Rasel  
**Research Team:**
- **Member 1:** Zannatun Nayem (Student ID: 23301681)
- **Member 2:** Farin Zaman Audri (Student ID: 23341041)
- **Member 3:** Abu Nayeem Md. Tanzim Adib (Student ID: 22301019)

---

## 1. Executive Summary

This research develops an end-to-end explainable artificial intelligence (XAI) and generative balancing framework for detecting **Diabetic Retinopathy (DR)** (5-class ordinal grading on EyePACS) and **Glaucoma** (binary screening on REFUGE2). To bridge the gap between deep black-box predictions and real clinical workflows, the system integrates **Grad-CAM / Grad-CAM++**, **quantitative explanation faithfulness tests**, and a **Retrieval-Augmented Generation (RAG)** clinical reporting layer referencing international guidelines (AAO, ICO, EGS, ETDRS).

---

## 2. Experimental Work Completed So Far

### 2.1 Dataset Ingestion & Auditing
- **EyePACS (`CDHAI/EyePACS` via Hugging Face Hub):**
  - Audited total distribution: ~143,669 fundus images partitioned across train, validation, and test splits.
  - Class distribution verified: Severe class imbalance observed, where **No DR (Grade 0)** represents ~73.5% of images, while **Severe DR (Grade 3)** and **Proliferative DR (Grade 4)** represent less than 5% combined.
- **Preprocessing Pipeline:**
  - Implemented standard image resolution pipeline (224×224 RGB).
  - Designed Contrast-Limited Adaptive Histogram Equalization (**CLAHE**) on the luminance channel to enhance microaneurysms and optic disc contours.
  - Implemented torchvision data transformations with geometric and color jitter augmentations.

### 2.2 Deep Learning Model Training & Empirical Baselines
Preliminary experimental trials were performed on GPU compute (Google Colab / PyTorch) as documented in preliminary runs:

1. **DenseNet-121 Pilot Run:**
   - **Data Split:** 3,000 training examples, 600 validation examples (EyePACS subset).
   - **Training Parameters:** 5 epochs, AdamW optimizer, Cross-Entropy loss.
   - **Validation Accuracy Progression:**
     - Epoch 1: 58.50%
     - Epoch 2: 69.00% (Best checkpoint saved: `dr_baseline_densenet121.pt`)
     - Epoch 3–5: 63.83% – 66.17% (evidence of overfitting on small subset without focal loss).

2. **EfficientNet-B0 Scaled Pilot Run:**
   - **Data Split:** 15,000 training examples, 3,000 validation examples.
   - **Training Parameters:** 10 epochs, learning rate $2.5 \times 10^{-5}$, Cosine Annealing scheduler.
   - **Validation Performance:**
     - Epoch 1: 37.03% running accuracy
     - Epoch 5: 68.42% accuracy
     - Epoch 10: **75.80% validation accuracy** (Best checkpoint saved: `dr_v2_efficientnet.pt`).

### 2.3 Explainable AI (XAI) Implementations
- Integrated `pytorch-grad-cam` targeting the final convolutional features (`features.denseblock4.denselayer16.conv2` in DenseNet121 and `features[-1]` in EfficientNet-B0).
- Successfully generated sample class activation overlay (`gradcam_output_efficientnet.png`) for a verified Proliferative DR fundus image, demonstrating focal attention on pathological vascular abnormalities.
- Implemented quantitative faithfulness deletion curve evaluation to measure prediction probability degradation under progressive masking.
- Implemented Adebayo model-randomization sanity check logic.

### 2.4 Generative Adversarial Network (GAN) Design
- Designed Conditional GAN (`cGAN`) architecture capable of conditioned generation for minority classes (Mild, Moderate, Severe, Proliferative DR).
- Formulated generator and discriminator networks using transposed convolution blocks and class embedding concatenation.

### 2.5 Retrieval-Augmented Generation (RAG) Clinical Engine
- Curated ophthalmic clinical knowledge base JSON covering ETDRS 4-2-1 rule, ICO, AAO, and EGS guidelines.
- Developed semantic retriever with fallback keyword indexing.
- Implemented automated report generator creating structured decision-support summaries.

---

## 3. Literature Review Baselines & Target Metrics

In direct response to supervisor guidance (*"We have to surpass the accuracy that was already reported by the authors in our literature review part"*):

| Reference Study | Task & Dataset | Architecture | Baseline Result | Our Target Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Gulshan et al. (2016)** | Referable DR (EyePACS-1) | Inception-v3 | 99.1% AUC | Target **>99.2% AUC** via DenseNet121 + EfficientNet ensemble |
| **Ting et al. (2017)** | DR & Glaucoma (71k cohort) | Deep CNN | 93.6% AUC (DR), 94.2% AUC (Glaucoma) | Target **>95.5% AUC** on joint screening |
| **Pascal et al. (2022)** | Glaucoma (REFUGE) | MTL-IO (VGG-16 + U-Net) | 96.76% AUC | Target **>97.5% AUC** on REFUGE2 with disc attention |
| **Zhu et al. (2026)** | 5-Class DR (EyePACS + APTOS) | EfficientNet-B0 + DenseNet-121 | 91.2% Acc, 0.961 AUC | Target **>92.5% Acc** with cGAN oversampling & Focal Loss |
| **Al-Dolat et al. (2026)** | 5-Class DR (APTOS / DDR) | DenseNet-121 + CBAM | 96.12% Acc | Target **>96.5% Acc** with external cross-testing |

---

## 4. Remaining Work & 3-Member Task Division

To ensure rapid, rigorous completion, the remaining research tasks are divided equally among the three team members:

```
                               ┌──────────────────────────────────────────────────────────┐
                               │  Thesis: Explainable Retinal Disease Detection Framework │
                               └────────────────────────────┬─────────────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────┼─────────────────────────────────────────────────┐
         │                                                  │                                                 │
┌────────▼────────────────────────┐       ┌─────────────────▼───────────────┐       ┌─────────────────────────▼──────────────┐
│ Member 1: Zannatun Nayem        │       │ Member 2: Farin Zaman Audri     │       │ Member 3: Abu Nayeem Md. Tanzim Adib   │
│ DR Models & GAN Augmentation    │       │ Glaucoma Model & Disc Analysis  │       │ XAI Verification & Clinical RAG Engine │
└─────────────────────────────────┘       └─────────────────────────────────┘       └────────────────────────────────────────┘
```

### 4.1 Member 1: Zannatun Nayem (Student ID: 23301681)
**Focus Area:** Diabetic Retinopathy Classification & Generative Augmentation
- [ ] Scale EyePACS 5-class training from pilot subsets to the full training split.
- [ ] Train conditional GAN on minority classes (Grades 1, 2, 3, 4) and generate synthetic balanced datasets.
- [ ] Evaluate classification performance with:
  1. Baseline (unbalanced)
  2. Class-weighted / Focal Loss
  3. GAN-augmented training sets
- [ ] Benchmark single-model vs. Hybrid Weighted Ensemble (DenseNet-121 + EfficientNet-B0) to achieve target $>92.5\%$ 5-class accuracy and $>99.2\%$ referable AUC.
- [ ] Conduct external validation on the held-out **APTOS 2019** dataset.

### 4.2 Member 2: Farin Zaman Audri (Student ID: 23341041)
**Focus Area:** Glaucoma Classification & Optic Disc / Cup Analysis
- [ ] Setup and preprocess the **REFUGE2** dataset (2,000 images) and extract optic disc regions.
- [ ] Train deep transfer learning models (DenseNet-121 vs. EfficientNet-B0) for binary glaucoma detection.
- [ ] Implement optic disc mask alignment to compute the **percentage of Grad-CAM activation mass located within the optic disc mask**.
- [ ] Optimize classification thresholds to surpass Pascal et al.'s 96.76% AUC baseline on REFUGE2.
- [ ] Perform external generalizability validation using the **G1020** dataset.
- [ ] Compile multi-metric comparison tables (Accuracy, Sensitivity, Specificity, AUC, F1-score).

### 4.3 Member 3: Abu Nayeem Md. Tanzim Adib (Student ID: 22301019)
**Focus Area:** Explainable AI (XAI) Faithfulness & RAG Clinical Report Engine
- [ ] Implement and compare **Grad-CAM vs. Grad-CAM++** attributions across all 5 DR grades and glaucoma cases.
- [ ] Execute quantitative faithfulness deletion tests across cohorts and calculate the Area Under the Deletion Curve (AUDC).
- [ ] Run Adebayo model-randomization sanity checks to formally prove heatmaps rely on trained weights rather than edge detection heuristics.
- [ ] Finalize the vector database for retinal clinical guidelines (ETDRS, AAO, EGS, ICO) with `sentence-transformers`.
- [ ] Build end-to-end RAG clinical report generation pipeline linking model predictions, Grad-CAM findings, and clinical pathways.
- [ ] Quantitatively evaluate generated clinical reports using **ROUGE-L** and **SBERT similarity**.

---

## 5. Summary of Milestones & Submission Timeline

| Phase | Milestone Objective | Target Completion | Responsible Member(s) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Repository initialization, baseline pilots & git commit | **Completed** | Full Team |
| **Phase 2** | Full EyePACS & REFUGE2 model convergence | Week 1 | Member 1 & Member 2 |
| **Phase 3** | cGAN minority generation & disc mask XAI evaluation | Week 2 | Member 1 & Member 2 |
| **Phase 4** | XAI faithfulness curves & RAG vector store completion | Week 3 | Member 3 |
| **Phase 5** | External testing (APTOS 2019 + G1020) & metric benchmarking | Week 4 | Member 1 & Member 2 |
| **Phase 6** | ROUGE-L report validation & final thesis chapter synthesis | Week 5 | Full Team |
