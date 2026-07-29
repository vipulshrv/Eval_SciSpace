# Deep Learning for Diabetic Retinopathy Screening from Retinal Fundus Images: A Comparative Analysis of CNN Architectures, Sensitivity, Specificity, and AUC Across Datasets

## Executive Summary

Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide, affecting over 93 million individuals. Automated screening using deep learning, particularly convolutional neural networks (CNNs), has emerged as a transformative approach to scaling DR detection beyond the constraints of specialist ophthalmologic assessment. This report synthesizes findings from recent studies (2016–2026) that evaluate CNN-based models for DR screening from retinal fundus images, comparing their diagnostic performance in terms of sensitivity, specificity, and area under the receiver operating characteristic curve (AUC).

The evidence demonstrates that modern CNN architectures—including EfficientNet, ResNet, DenseNet, Inception, and hybrid CNN-Transformer models—achieve clinically relevant performance, with sensitivity values ranging from 74% to 99%, specificity from 87% to 99%, and AUC from 91% to 99% depending on the architecture, dataset, and classification task (binary vs. multi-class severity grading). Models trained on large-scale datasets such as EyePACS and validated externally on APTOS, Messidor, and IDRiD show strong but variable generalization. Hybrid architectures and attention mechanisms represent the current frontier, offering incremental improvements over classical CNNs while enhancing interpretability. Key challenges remain in cross-dataset generalization, class imbalance, and deployment in resource-limited clinical settings.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Theoretical Foundations](#2-background-and-theoretical-foundations)
3. [CNN Architectures for Diabetic Retinopathy Detection](#3-cnn-architectures-for-diabetic-retinopathy-detection)
   - 3.1 [Classical CNN Architectures](#31-classical-cnn-architectures)
   - 3.2 [EfficientNet Family](#32-efficientnet-family)
   - 3.3 [Hybrid and Ensemble Models](#33-hybrid-and-ensemble-models)
   - 3.4 [CNN-Transformer Fusion Architectures](#34-cnn-transformer-fusion-architectures)
   - 3.5 [Attention-Enhanced Models](#35-attention-enhanced-models)
4. [Benchmark Datasets](#4-benchmark-datasets)
5. [Comparative Performance Analysis](#5-comparative-performance-analysis)
   - 5.1 [Performance Summary Table](#51-performance-summary-table)
   - 5.2 [Sensitivity Analysis](#52-sensitivity-analysis)
   - 5.3 [Specificity Analysis](#53-specificity-analysis)
   - 5.4 [AUC Analysis](#54-auc-analysis)
   - 5.5 [Impact of Dataset and Task Complexity](#55-impact-of-dataset-and-task-complexity)
6. [Discussion](#6-discussion)
   - 6.1 [Architecture-Performance Relationships](#61-architecture-performance-relationships)
   - 6.2 [Cross-Dataset Generalization](#62-cross-dataset-generalization)
   - 6.3 [Binary vs. Multi-Class Classification](#63-binary-vs-multi-class-classification)
   - 6.4 [Limitations and Methodological Considerations](#64-limitations-and-methodological-considerations)
7. [Future Directions and Recommendations](#7-future-directions-and-recommendations)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

Diabetic retinopathy is a microvascular complication of diabetes mellitus that affects the retinal vasculature, leading to progressive vision loss if undetected and untreated. With the global diabetic population exceeding 500 million, the demand for scalable, accurate, and accessible screening programs has intensified. Traditional DR screening relies on manual grading of retinal fundus photographs by ophthalmologists—a process that is time-consuming, costly, and subject to inter-observer variability [3].

The advent of deep learning, particularly convolutional neural networks, has provided a paradigm shift in medical image analysis. Since the landmark study by Gulshan et al. (2016), which demonstrated that a deep learning algorithm could detect referable DR with high sensitivity and specificity from fundus images [27], the field has rapidly expanded. Numerous architectures—from AlexNet and VGGNet to modern EfficientNets, DenseNets, and hybrid CNN-Transformer models—have been evaluated across diverse datasets, classification tasks, and clinical settings [20], [24].

This report provides a systematic comparative analysis of CNN architectures for DR screening, focusing on three critical diagnostic metrics: sensitivity (the ability to correctly identify DR cases), specificity (the ability to correctly identify healthy cases), and AUC (the overall discriminative ability of the model). The analysis draws on studies utilizing major benchmark datasets including EyePACS, APTOS, Messidor, IDRiD, and various clinical datasets, enabling assessment of both internal validation performance and cross-dataset generalizability.

---

## 2. Background and Theoretical Foundations

### 2.1 Diabetic Retinopathy Grading

DR is clinically graded on a severity scale: No DR, Mild Non-Proliferative DR (NPDR), Moderate NPDR, Severe NPDR, and Proliferative DR (PDR). Screening systems may perform binary classification (referable vs. non-referable DR) or multi-class severity grading across all five levels. The choice of classification granularity directly impacts reported performance metrics, with binary tasks generally yielding higher sensitivity and specificity values than five-class grading [3], [5].

### 2.2 Deep Learning in Medical Imaging

CNNs extract hierarchical spatial features from images through successive convolutional, pooling, and fully connected layers. Transfer learning—pre-training on large-scale natural image datasets (e.g., ImageNet) and fine-tuning on medical images—has become the standard paradigm for DR detection, addressing the challenge of limited labeled medical data [20], [28]. Key architectural innovations include residual connections (ResNet), dense connectivity (DenseNet), efficient channel scaling (EfficientNet), inception modules (GoogLeNet/Inception), and more recently, attention mechanisms and transformer-based global context modeling [7], [21].

### 2.3 Performance Metrics in DR Screening

Three primary metrics are used to evaluate DR screening systems:

- **Sensitivity (Recall):** The proportion of actual DR cases correctly identified. High sensitivity is critical for screening to minimize missed diagnoses.
- **Specificity:** The proportion of non-DR cases correctly identified. High specificity reduces unnecessary referrals.
- **AUC (Area Under the ROC Curve):** A threshold-independent measure of a model's ability to discriminate between DR and non-DR across all operating points, with values closer to 1.0 indicating better discrimination [3], [24].

---

## 3. CNN Architectures for Diabetic Retinopathy Detection

### 3.1 Classical CNN Architectures

**ResNet (Residual Networks):** ResNet architectures, particularly ResNet-50 and ResNet-101, are among the most frequently employed models for DR detection. Zaier and Zribi (2025) reported that ResNet-50 achieved 96% accuracy, 94% sensitivity, 97% specificity, and an AUC of 96% on external validation using the APTOS dataset [1]. Asia et al. (2022) demonstrated that ResNet-101 achieved 97% accuracy, 96.87% sensitivity, 98.03% specificity, and an AUC of 97.26% across multiple datasets [22]. The residual learning framework enables training of very deep networks without degradation, making it well-suited for fine-grained DR feature extraction.

**VGGNet:** VGG-16 and VGG-19 remain popular baseline architectures. Çinarer and Kiliç (2021) reported VGG-16 achieving a 99.6% AUC and 98.1% accuracy on the APTOS dataset [11]. The survey by Sarki et al. (2020) noted VGGNet achieving 93.6% AUC, 90.5% sensitivity, and 91.6% specificity for DR detection [20].

**AlexNet:** As an earlier architecture, AlexNet provides a useful baseline. Elsawah et al. (2024) achieved 95.73% accuracy, 95.73% sensitivity, and 98.51% specificity using an AlexNet-based system on the IDRiD dataset [18]. Sarki et al. (2020) reported AlexNet achieving 98.0% AUC, 96.8% sensitivity, and 87.0% specificity on Messidor-2 [20].

**Inception (GoogLeNet) Family:** InceptionV3 and InceptionResNet incorporate multi-scale feature extraction through parallel convolutional pathways. Reguant et al. (2021) evaluated Inception-v3, ResNet50, InceptionResNet, and Xception on EyePACS and DIARETDB1, finding accuracy of 89–95%, AUC of 95–98%, sensitivity of 74–86%, and specificity of 93–97% [4].

### 3.2 EfficientNet Family

EfficientNet architectures use compound scaling to balance network depth, width, and input resolution, achieving high performance with fewer parameters. EfficientNet-B5 achieved 98% sensitivity, 93% specificity, and 96% AUC for DR detection, showing particularly strong agreement scores (QWK 92%) on external validation [1]. EfficientNetV2B0 achieved an AUC of 0.93 while offering significant reductions in energy consumption (66%) and training time (70%), making it suitable for resource-constrained deployments [12]. The EfficientNet family has been utilized with attention mechanisms, where EfficientNet-b0 achieved 80% accuracy on both APTOS and EyePACS for multi-class severity grading [10].

### 3.3 Hybrid and Ensemble Models

Hybrid approaches combine features from multiple CNN backbones to leverage complementary representations. The IR-CNN model by the authors of [8] fuses InceptionV3 and ResNet50 features, achieving 96.85% accuracy, 99.28% sensitivity, and 98.92% specificity on a large fundus dataset with data augmentation [8]. Kumar et al. (2025) proposed a framework integrating ResNet, DenseNet, and Inception with attention-based feature fusion, achieving 97.85% accuracy, 95.68% recall, 98.12% specificity, and an AUC of 98.45% [9].

Ensemble methods, such as the stacked ensemble of ResNet18, ResNet50, and EfficientNetB3, demonstrated an ROC AUC of 0.933 on the APTOS dataset, with ResNet50 as the strongest individual contributor at 83.20% accuracy [6]. Lakhera et al. (2023) combined AlexNet, VGGNet, and ResNet-18 with AdaBoost on APTOS and Messidor-2 datasets, demonstrating the value of ensemble diversity in improving robustness [16].

### 3.4 CNN-Transformer Fusion Architectures

Recent research has explored integrating CNN feature extractors with transformer-based global attention. The DR-CTFN model by Huang et al. (2025) fuses ConvNeXt and Swin Transformer, outperforming ConvNeXt alone by 3.14% in accuracy and achieving AUC values of 95.22% and 95.79% on APTOS and clinical datasets, respectively [21]. Wong et al. (2025) compared classic CNNs with hybrid models, finding that MaxViT achieved a 91.8% AUC (95% CI: 90.5–93.1%), outperforming pure CNNs by 1.5–2.9% while reducing cross-dataset performance degradation by 38% [7]. Alooghareh et al. (2025) found that the Swin-L vision transformer achieved an AUC of 0.98 for DR classification on the BRSET dataset [13].

### 3.5 Attention-Enhanced Models

Attention mechanisms have been integrated into CNN backbones to improve lesion-specific feature extraction. Hannan et al. (2025) employed a dual-attention mechanism with MobileNetV3-small, EfficientNet-b0, and DenseNet-169, achieving 83.0% sensitivity, 95.5% specificity, and a kappa score of 88.2% [10]. Kaur et al. (2025) compared CBAM, BAM, and ECA attention modules with ResNet50 on APTOS, finding that BAM yielded the best experimental results for multi-class DR classification [26]. Shailee (2025) reported that a modified DenseNet-121 achieved 99.1% sensitivity and 98.6% specificity for distinguishing No DR in smartphone-based fundus imaging [17].

---

## 4. Benchmark Datasets

The choice of dataset profoundly influences reported performance metrics. Key datasets used in DR screening research include:

| Dataset | Size | Classes | Source | Notable Usage |
|---------|------|---------|--------|---------------|
| EyePACS (Kaggle) | ~88,000 images | 5 DR grades | US telemedicine program | Most widely used for training [1], [24] |
| APTOS 2019 | 3,662 images | 5 DR grades | Asia Pacific screening | Common external validation [1], [6] |
| Messidor / Messidor-2 | 1,200 / 1,748 images | 4 retinopathy grades | French screening program | Cross-dataset validation [7], [20] |
| IDRiD | 516 images | 5 DR grades | Indian clinical data | Segmentation and grading [18] |
| DIARETDB0/1 | 130 / 89 images | Binary | Finnish clinical data | Small-scale validation [4] |
| DDR | ~13,000 images | 6 grades | Chinese clinical data | Grading benchmark [14] |
| BRSET | 16,266 images | Multi-label | Brazilian clinical data | Multi-condition detection [13], [19] |

Studies that train on EyePACS and validate externally on APTOS or Messidor provide the most clinically relevant generalization evidence. Smaller datasets such as IDRiD and DIARETDB serve primarily for specialized evaluations. The systematic review by Alshammari et al. (2025) confirmed that EyePACS was used by the majority of evaluated works, with cross-validation on Messidor and IDRiD being common practice [24].

---

## 5. Comparative Performance Analysis

### 5.1 Performance Summary Table

The following table summarizes key performance metrics across architectures and datasets from the reviewed studies:

| Study | Architecture | Dataset(s) | Sensitivity | Specificity | AUC | Task |
|-------|-------------|------------|-------------|-------------|-----|------|
| Zaier & Zribi (2025) [1] | EfficientNet-B5 | EyePACS → APTOS | 98% | 93% | 96% | Multi-class |
| Zaier & Zribi (2025) [1] | ResNet-50 | EyePACS → APTOS | 94% | 97% | 96% | Multi-class |
| Sirisati et al. (2025) [2] | ResNet50 / InceptionV3 | EyePACS | 99% | 99% | — | Multi-class |
| Reguant et al. (2021) [4] | Inception-v3, ResNet50, Xception | EyePACS, DIARETDB1 | 74–86% | 93–97% | 95–98% | Multi-class |
| Wong et al. (2025) [7] | MaxViT (hybrid) | EyePACS → APTOS, Messidor | — | — | 91.8% | Multi-class |
| IR-CNN (2023) [8] | InceptionV3 + ResNet50 | Kaggle (44,119 images) | 99.28% | 98.92% | — | Multi-class |
| Kumar et al. (2025) [9] | ResNet + DenseNet + Inception (fusion) | Benchmark | 95.68%* | 98.12% | 98.45% | Binary |
| Hannan et al. (2025) [10] | DenseNet-169 / EfficientNet-b0 | APTOS, EyePACS | 83.0% | 95.5% | — | Multi-class |
| Çinarer & Kiliç (2021) [11] | VGG-16 | APTOS 2019 | — | — | 99.6% | Multi-class |
| Araújo et al. (2025) [12] | EfficientNetV2B0 | UNIFESP (Brazilian) | — | — | 93% | Binary |
| Alooghareh et al. (2025) [13] | Swin-L | BRSET | — | — | 98% | Binary/3-class |
| Zeng et al. (2019) [15] | Siamese CNN | Private (28,104 images) | 80.7%† | 95.0%† | 94.9% | Binary |
| Shailee (2025) [17] | DenseNet-121 | APTOS (Kaggle) | 99.1% | 98.6% | — | Multi-class |
| Elsawah et al. (2024) [18] | AlexNet-based | IDRiD | 95.73% | 98.51% | — | Multi-class |
| Sarki et al. (2020) [20] | AlexNet | Messidor-2 | 96.8% | 87.0% | 98.0% | Binary |
| Sarki et al. (2020) [20] | VGGNet | Multiple | 90.5% | 91.6% | 93.6% | Binary |
| Huang et al. (2025) [21] | DR-CTFN (ConvNeXt + Swin) | EyePACS → APTOS | — | — | 95.22% | Multi-class |
| Asia et al. (2022) [22] | ResNet-101 | XHO, MESSIDOR, HRF | 96.87% | 98.03% | 97.26% | Binary |
| Son (2023) [23] | DeepPCANet-4 | APTOS 2019 | 95.29% | 98.9% | — | Multi-class |

*Reported as recall; †At fixed operating points.

### 5.2 Sensitivity Analysis

Sensitivity represents the most critical metric for DR screening, as missed cases can lead to irreversible vision loss. Across the reviewed literature, sensitivity values demonstrate substantial variation by architecture and task complexity:

- **Highest sensitivity (≥98%):** EfficientNet-B5 achieved 98% sensitivity [1], while hybrid models combining InceptionV3 and ResNet50 reported 99.28% [8]. The ResNet50/InceptionV3 transfer learning approach also reported 99% sensitivity [2]. DenseNet-121 achieved 99.1% sensitivity for No DR classification [17].
- **Moderate sensitivity (85–97%):** ResNet-50 alone achieved 94–96.87% [1], [22], AlexNet-based systems reached 95.73–96.8% [18], [20], and automatically designed CNN architectures (DeepPCANet) attained 95.29% [23].
- **Lower sensitivity (74–85%):** Multi-class grading with Inception-v3 and related architectures on complex datasets showed sensitivity values as low as 74–86% [4], and dual-attention models reported 83.0% when evaluated on EyePACS [10].

The observed pattern indicates that sensitivity tends to be higher for binary classification (referable vs. non-referable) and decreases for five-class severity grading. Hybrid and ensemble models generally achieve higher sensitivity than individual architectures operating in isolation.

### 5.3 Specificity Analysis

High specificity is important to minimize false positive referrals, which place unnecessary burden on ophthalmology services. Key observations include:

- **Highest specificity (≥98%):** ResNet-50 achieved 97% specificity [1], hybrid IR-CNN reported 98.92% [8], the attention-fusion framework reached 98.12% [9], ResNet-101 attained 98.03% [22], and AlexNet-based systems on IDRiD achieved 98.51% [18]. DenseNet-121 on APTOS reported 98.6% [17]. The DeepPCANet-4 architecture achieved 98.9% specificity on APTOS [23].
- **Moderate specificity (91–97%):** EfficientNet-B5 reported 93% [1], Inception-v3 and Xception ranged from 93–97% [4], dual-attention DenseNet-169 reached 95.5% [10], and VGGNet reported 91.6% [20].
- **Lower specificity (<91%):** AlexNet on Messidor-2 showed 87.0% specificity [20], reflecting the architecture's limited capacity for fine-grained discrimination.

Specificity values are generally high across modern architectures (>93%), with ResNet variants and hybrid models consistently exceeding 97%. The trade-off between sensitivity and specificity is evident: EfficientNet-B5 optimizes for sensitivity (98%) with somewhat lower specificity (93%), while ResNet-50 shows the inverse pattern (94% sensitivity, 97% specificity) [1].

### 5.4 AUC Analysis

AUC provides a threshold-independent assessment of overall discriminative performance:

- **Highest AUC (≥97%):** VGG-16 on APTOS achieved 99.6% [11], the attention-fusion framework reached 98.45% [9], AlexNet on Messidor-2 reported 98.0% [20], Swin-L transformer achieved 98% [13], Inception-v3/ResNet50 group ranged from 95–98% [4], and ResNet-101 achieved 97.26% across multiple datasets [22].
- **Moderate AUC (93–97%):** EfficientNet-B5 and ResNet-50 both achieved 96% on external validation [1], DR-CTFN (CNN-Transformer fusion) reported 95.22–95.79% [21], the Siamese CNN reached 94.9% [15], and EfficientNetV2B0 achieved 93% [12].
- **Lower AUC (91–93%):** MaxViT hybrid achieved 91.8% [7], and MobileNet reported 91% [12] in resource-efficient configurations.

The AUC evidence suggests that even lightweight models can achieve clinically acceptable discrimination (>90%), while deeper or hybrid architectures approach near-perfect discrimination on well-curated datasets. However, AUC values consistently decrease (by 2–5%) when models are evaluated on external datasets compared to internal validation.

### 5.5 Impact of Dataset and Task Complexity

Dataset characteristics significantly influence reported performance. Models trained on large-scale datasets (EyePACS, ~88,000 images) and tested on the same distribution typically report higher metrics than those evaluated cross-dataset. Wong et al. (2025) quantified this effect, showing that ResNet50 suffered a 4.2% performance drop in cross-dataset evaluation, whereas hybrid MaxViT models reduced this gap to 2.5% [7].

Multi-class severity grading (five classes) consistently yields lower performance than binary classification. For instance, DeepPCANet achieved 99.5% sensitivity in binary mode versus 95.29% for five-class grading on the same dataset family [23]. This pattern is replicated across the literature, with binary classification typically yielding 3–10% higher sensitivity and specificity values compared to multi-class grading [4], [14].

Class imbalance—where mild and moderate DR cases are underrepresented—further complicates multi-class performance. Data augmentation, class weighting, and undersampling strategies have been employed to mitigate this effect [8], [17].

---

## 6. Discussion

### 6.1 Architecture-Performance Relationships

The comparative evidence reveals several patterns in architecture-performance relationships. ResNet variants (ResNet-50, ResNet-101) provide a strong balance of sensitivity, specificity, and AUC while being computationally tractable, making them the most popular backbone for DR detection [1], [22], [25]. EfficientNet architectures offer competitive performance with improved computational efficiency, particularly advantageous for deployment in resource-constrained environments such as mobile screening devices [12], [17].

Hybrid models that combine feature extractors from different CNN families (e.g., Inception + ResNet, ConvNeXt + Swin Transformer) consistently outperform their individual components, suggesting that architectural diversity in feature extraction provides complementary information beneficial for DR classification [8], [9], [21]. However, these gains come at the cost of increased model complexity and training requirements.

### 6.2 Cross-Dataset Generalization

A critical challenge for clinical deployment is cross-dataset generalization. Models trained exclusively on EyePACS may not perform equivalently on images from different camera systems, populations, or clinical settings. The evidence shows that hybrid CNN-Transformer architectures demonstrate superior robustness to domain shift, with MaxViT reducing cross-dataset performance dips by 38% relative to pure CNNs [7]. External validation studies—such as training on EyePACS and testing on APTOS—provide the most realistic estimates of clinical performance [1], [21].

The systematic review by Alshammari et al. (2025) noted that while CNN-based models consistently achieve accuracy above 90% on internal validation, performance on external datasets and in real clinical workflows can be lower due to variations in image quality, camera type, and patient demographics [24].

### 6.3 Binary vs. Multi-Class Classification

The distinction between binary (referable vs. non-referable) and multi-class (five severity grades) classification is fundamental to interpreting performance comparisons. Binary classification is more clinically aligned with screening programs, where the primary goal is identifying patients requiring specialist referral. Multi-class grading provides finer clinical granularity but is significantly more challenging due to subtle inter-class differences, particularly between mild and moderate NPDR stages [4], [5].

Studies reporting very high metrics (>98% sensitivity, >98% specificity) often employ binary classification or evaluate specific severity boundaries rather than full five-class grading [2], [8]. This must be considered when comparing results across studies.

### 6.4 Limitations and Methodological Considerations

Several limitations affect the interpretability of reported performance across studies:

1. **Inconsistent metric reporting:** Not all studies report the same set of metrics. Some report only accuracy or AUC without sensitivity and specificity, making direct comparison difficult [5], [16], [26].
2. **Dataset overlap and leakage:** Many studies use the same public datasets (EyePACS, APTOS), raising concerns about indirect data leakage through overlapping preprocessing pipelines or architectures optimized for specific dataset characteristics [24].
3. **Preprocessing variability:** Different studies apply varying preprocessing techniques (CLAHE, cropping, resizing, augmentation), which can significantly impact performance and limit reproducibility [26], [6].
4. **Limited real-world validation:** Most studies evaluate on curated datasets rather than prospective clinical cohorts, which may overestimate real-world screening performance [24].
5. **Class imbalance:** The uneven distribution of DR severity levels in most datasets particularly affects sensitivity for mild DR, which is the most important stage for early intervention [3], [10].

---

## 7. Future Directions and Recommendations

Based on the reviewed evidence, several directions are recommended for advancing deep learning-based DR screening:

1. **Standardized benchmarking:** The field would benefit from standardized evaluation protocols that consistently report sensitivity, specificity, and AUC at predefined operating points, alongside confidence intervals, to facilitate fair comparison across architectures [3], [24].

2. **Cross-dataset and prospective validation:** Future studies should prioritize multi-center, prospective validation to assess real-world clinical utility rather than relying solely on retrospective evaluations on public datasets [1], [7].

3. **Efficient architectures for deployment:** Models such as EfficientNet and MobileNet that balance performance with computational efficiency are particularly relevant for deployment in resource-limited settings through smartphone-based screening [12], [17].

4. **Hybrid architectures and attention mechanisms:** CNN-Transformer fusion models and attention-enhanced architectures represent promising directions for improving both performance and interpretability [21], [10], [26].

5. **Explainability and clinical trust:** Integration of interpretability tools such as Grad-CAM, SHAP, and attention visualization is essential for clinical adoption, enabling ophthalmologists to understand and verify model decisions [2], [6].

6. **Addressing class imbalance:** Advanced augmentation, synthetic data generation, and cost-sensitive learning strategies should be further explored to improve detection of mild and early-stage DR [8], [14].

---

## 8. Conclusion

Deep learning-based approaches for diabetic retinopathy screening from retinal fundus images have achieved performance levels approaching or exceeding those of expert ophthalmologists across multiple metrics. The comparative analysis reveals that:

- **ResNet-50/101** provides consistently strong and well-balanced performance (sensitivity 94–97%, specificity 97–98%, AUC 96–97%), serving as a reliable baseline architecture [1], [22].
- **EfficientNet** variants achieve high sensitivity (up to 98%) with improved computational efficiency, suitable for scalable screening programs [1], [12].
- **Hybrid CNN models** combining Inception and ResNet features yield the highest reported sensitivity (99.28%) and specificity (98.92%) in controlled evaluations [8], [9].
- **CNN-Transformer fusion** architectures represent the current frontier, offering superior AUC (95–98%) and improved cross-dataset robustness [7], [13], [21].
- **DenseNet-121** with attention mechanisms demonstrates exceptional performance in smartphone-based screening scenarios [17].

Performance varies substantially with task complexity (binary vs. multi-class), dataset characteristics, and evaluation methodology. The most clinically meaningful evaluations are those performing external validation across different datasets and populations. While the field has made remarkable progress, the path to widespread clinical deployment requires continued focus on generalization, standardized reporting, computational efficiency, and integration within existing healthcare workflows.

