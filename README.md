# LW5_Comparative-Analysis-of-Pre-trained-CNN-Models_CSC120

GDRIVE link: https://drive.google.com/drive/folders/19oy7WYlHUFdy3h8TCm34t7I_FQAkSr_o?usp=drive_link

COLAB Link: https://colab.research.google.com/drive/115525Kdk9nbUS31SxZvC-zoQfal9qBJa?usp=sharing

---

# Comparative Analysis of Pre-trained CNN Models for Custom Image Classification

## Pre-trained Models Used
1. **MobileNetV2** — Fastest, best for mobile deployment
2. **EfficientNetB0** — Best overall accuracy
3. **InceptionV3** — Stable, strong baseline

## Dataset
- **20 Mushroom Species** — 250 images per class
- **Total Images:** 5,000
- **Split:** 80% Training / 20% Validation
- **Image Size:** 224×224 pixels

---

## Performance Comparison Table

### LW5 Pre-trained Models

| Model | Train Acc | Train Loss | Val Acc | Val Loss | Precision | Recall | F1-Score | AUC |
|-------|-----------|------------|---------|----------|-----------|--------|----------|-----|
| MobileNetV2 | 96.70% | 0.1167 | **97.70%** | **0.0596** | **98.47%** | **98.05%** | **97.88%** | **0.9989** |
| EfficientNetB0 | 96.78% | 0.1075 | 97.30% | 0.0663 | 98.29% | 97.71% | 97.49% | 0.9977 |
| InceptionV3 | 88.75% | 0.5236 | 94.50% | 0.1500 | 94.87% | 94.74% | 94.64% | 0.9961 |

### Full Comparison Table (All Models — All Lab Works)

| Model | Val Accuracy | Val Loss | Precision | Recall | F1-Score | AUC | Notes |
|-------|-------------|----------|-----------|--------|----------|-----|-------|
| Teachable Machine | 24.40% | — | 20.80% | 24.19% | 22.00% | 0.6338 | Label order mismatch |
| LW3 — Custom CNN | 93.20% | 0.2215 | 94.91% | 93.18% | 92.89% | 0.9644 | Baseline improved model |
| LW4 — Enhanced CNN | 96.70% | 0.0644 | 97.80% | 96.85% | 96.68% | 0.9963 | Best custom CNN |
| LW5 — InceptionV3 | 94.50% | 0.1500 | 94.87% | 94.74% | 94.64% | 0.9961 | Pre-trained (ImageNet) |
| LW5 — EfficientNetB0 | 97.30% | 0.0663 | 98.29% | 97.71% | 97.49% | 0.9977 | Pre-trained (ImageNet) |
| **LW5 — MobileNetV2** | **97.70%** | **0.0596** | **98.47%** | **98.05%** | **97.88%** | **0.9989** | 🏆 Best overall |

---

# Guide Questions (Final Reflection)

## A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**

Answer: **MobileNetV2** achieved the highest validation accuracy at **97.70%** with a validation loss of **0.0596**. MobileNetV2 performed best because its architecture uses depthwise separable convolutions — a highly efficient design that extracts rich spatial features while maintaining a compact parameter count. This efficiency allowed the model to transfer ImageNet-learned features to the mushroom dataset more effectively within the 10-epoch training limit. MobileNetV2's lightweight design also meant it was less prone to overfitting on our 4,000-image training set compared to heavier architectures, and its strong feature extraction backbone generalized well across all 20 mushroom species — from the visually distinctive *Hericium erinaceus* to the more subtle differences between the four *Agaricus* species.

**2. Which model had the lowest performance? What could be the reason?**

Answer: The **Teachable Machine** model had the lowest performance with only **24.40% validation accuracy** and an AUC score of **0.6338**. The primary reason was a **class label order mismatch** — Teachable Machine organized the 20 mushroom classes in a different order than TensorFlow's `image_dataset_from_directory()`, which loads classes alphabetically. This caused predictions to be matched against the wrong class names, resulting in most classes scoring 0% precision and recall. Among the three pre-trained models, **InceptionV3** had the lowest performance at **94.50%** due to EarlyStopping triggering after only 1 effective training epoch — meaning it had insufficient time to fine-tune its classification head for the mushroom dataset despite its powerful pretrained features.

**3. How did loss values compare across models?**

Answer: Training loss values showed a clear pattern across the three pre-trained models. MobileNetV2 achieved a training loss of **0.1167**, EfficientNetB0 **0.1075**, and InceptionV3 **0.5236** — reflecting InceptionV3's limited training epochs. For validation loss, MobileNetV2 achieved **0.0596** (best), EfficientNetB0 **0.0663** (very close second), and InceptionV3 **0.1500** (highest). Comparing across all lab works, the progression of validation loss showed consistent improvement — LW3 baseline (0.2215) → LW4 enhanced (0.0644) → LW5 MobileNetV2 (0.0596) — demonstrating how each successive improvement brought the model closer to optimal performance. All three pre-trained models showed validation loss lower than training loss, indicating strong generalization with no overfitting.

---

## B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**

Answer: Accuracy alone is insufficient because it only measures the proportion of correct predictions without revealing how the model performs across individual classes. The Teachable Machine model is a perfect example — its 24.40% accuracy masked a critical underlying problem: a class label order mismatch that caused it to correctly identify only 4 out of 20 species while completely failing on the remaining 16. Even among well-performing models, accuracy differences can be misleading — the LW3 baseline (93.20% accuracy) had an AUC of only 0.9644, while the LW4 enhanced model (96.70% accuracy) achieved 0.9963 AUC, showing that the improvement in discriminative ability was proportionally greater than the accuracy improvement alone suggested. For a real-world mushroom safety application where missing a toxic species like *Amanita muscaria* or *Conocybe apala* could be life-threatening, accuracy alone would be a dangerously incomplete metric.

**5. Which model had the best F1-score? What does it indicate?**

Answer: **MobileNetV2** achieved the best macro F1-score at **97.88%**, followed by EfficientNetB0 at 97.49%, LW4 Enhanced CNN at 96.68%, InceptionV3 at 94.64%, LW3 baseline at 92.89%, and Teachable Machine at 22.00%. A high F1-score of 97.88% indicates that MobileNetV2 achieved an excellent balance between Precision (98.47%) and Recall (98.05%) across all 20 mushroom classes — meaning it both correctly identified mushrooms when it predicted a class and successfully captured the vast majority of actual instances of each species. The progression of F1-scores across lab works (92.89% → 96.68% → 97.88%) also confirms that each successive model improvement produced genuine gains in balanced classification performance, not just accuracy inflation.

**6. How did Precision and Recall differ across models?**

Answer: All three pre-trained models maintained closely matched Precision and Recall scores, indicating balanced classification. MobileNetV2 achieved Precision of **98.47%** and Recall of **98.05%** — a gap of only 0.42%. EfficientNetB0 showed Precision of **98.29%** and Recall of **97.71%** — a gap of 0.58%. InceptionV3 had Precision of **94.87%** and Recall of **94.74%** — a gap of 0.13%. The custom CNN models also showed this pattern — LW3 had Precision of **94.91%** and Recall of **93.18%** (gap of 1.73%), while LW4 improved to Precision of **97.80%** and Recall of **96.85%** (gap of 0.95%). The consistently higher Precision than Recall across all models suggests they were slightly conservative — when they predicted a class, they were very likely correct, but they occasionally missed some instances of visually similar species like the four *Agaricus* varieties.

---

## C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**

Answer: Based on the confusion matrix analysis, the most frequently misclassified classes were species within the same genus sharing similar morphological features. The four *Agaricus* species (*Agaricus arvensis*, *Agaricus augustus*, *Agaricus campestris*, and *Agaricus sylvicola*) showed the highest inter-class confusion due to their similar white-to-brown cap coloration, ring structures, and gill patterns. The two *Cortinarius* species (*Cortinarius mucosus* and *Cortinarius violaceus*) also showed occasional misclassification. In contrast, visually distinctive species like *Hericium erinaceus*, *Geastrum rufescens*, and *Auricularia auricula-judae* showed near-perfect classification across all models. The Teachable Machine model misclassified 16 out of 20 classes entirely due to the label order mismatch between its training configuration and TensorFlow's alphabetical class ordering.

**8. What patterns did you observe in the confusion matrix?**

Answer: The confusion matrices across all three pretrained models revealed two consistent patterns. First, **diagonal dominance** — the vast majority of predictions fell on the main diagonal, confirming high overall accuracy across all 20 species. Second, **intra-genus confusion** — the most significant off-diagonal values clustered among species of the same genus, particularly within *Agaricus* and *Cortinarius*, where visual similarity between species created the hardest classification boundaries. MobileNetV2 showed the cleanest confusion matrix with the fewest off-diagonal values, consistent with its highest validation accuracy of 97.70% and F1-score of 97.88%. InceptionV3 showed slightly more off-diagonal activity reflecting its limited fine-tuning. The Teachable Machine confusion matrix showed a completely different pattern — high values concentrated in only 4 diagonal positions with all other predictions scattered incorrectly due to the label mismatch.

---

## D. ROC and AUC

**9. Which model had the highest AUC score?**

Answer: **MobileNetV2** achieved the highest overall AUC score of **0.9989**, followed by EfficientNetB0 at **0.9977**, LW4 Enhanced CNN at **0.9963**, InceptionV3 at **0.9961**, LW3 baseline at **0.9644**, and Teachable Machine at **0.6338**. The progression of AUC scores across all lab works — from 0.9644 (LW3) to 0.9963 (LW4) to 0.9989 (LW5 MobileNetV2) — demonstrates consistent and meaningful improvement in discriminative ability with each model iteration. All three pretrained models achieved AUC scores above 0.99, placing them in the exceptional performance range and confirming that transfer learning from ImageNet features provided a significant advantage over training custom CNNs from scratch.

**10. What does AUC tell us about model performance?**

Answer: AUC (Area Under the ROC Curve) measures a model's ability to discriminate between classes across all possible classification thresholds — not just at the default threshold used for accuracy. For our 20-class mushroom classifier, AUC evaluates performance across all 190 possible class pairs simultaneously. Comparing AUC across all models reveals insights that accuracy alone misses — for example, the LW3 baseline (93.20% accuracy, AUC 0.9644) and InceptionV3 (94.50% accuracy, AUC 0.9961) have similar accuracy values but very different AUC scores, showing that InceptionV3's ImageNet-pretrained features gave it dramatically superior discriminative ability despite similar surface-level accuracy. The Teachable Machine's AUC of 0.6338 — barely above random guessing at 0.5 — confirmed that its poor accuracy was not just a label ordering artifact but reflected genuine difficulty in class discrimination.

---

## E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**

Answer: Grad-CAM revealed that all three pretrained models focused their attention primarily on the mushroom's cap and stem regions when making classification decisions — the most biologically meaningful parts of each specimen. The heatmap overlays showed concentrated activation on the distinctive morphological features of the test image rather than on the background or surrounding environment. This confirmed that the models were not exploiting spurious correlations in the training data but were instead learning genuine species-defining visual features from the ImageNet-pretrained feature extractors, fine-tuned to mushroom-specific characteristics. MobileNetV2 produced the most focused and precise heatmaps, consistent with its highest accuracy and AUC scores, while InceptionV3's heatmaps showed slightly broader activation regions reflecting its less refined fine-tuning from only 1 effective training epoch.

**12. Did the model focus on relevant image regions?**

Answer: Yes, all three models demonstrated focus on relevant mushroom regions in their Grad-CAM overlays. The heatmaps showed high activation concentrated on the mushroom body — particularly the cap surface, gill structure, and stem — rather than on the forest floor background or surrounding vegetation. This biologically meaningful focus is consistent with the strong validation accuracy achieved by all three models. The fact that MobileNetV2 predicted *Geastrum rufescens* and both EfficientNetB0 and InceptionV3 predicted *Craterellus tubaeformis* with moderate confidence scores (6–15%) — while their heatmaps remained focused on the mushroom body — suggests the test image contained visual features shared between these species, indicating genuine model uncertainty rather than irrelevant background exploitation.

**13. Which model produced the most meaningful heatmaps?**

Answer: **MobileNetV2** produced the most meaningful and focused Grad-CAM heatmaps, with activation concentrated tightly on the most distinctive morphological features of the mushroom specimen. This is consistent with MobileNetV2's depthwise separable convolution architecture, which learns highly specialized spatial filters that activate precisely on relevant features — producing cleaner gradient signals for Grad-CAM visualization. EfficientNetB0 produced similarly meaningful heatmaps with good localization on the mushroom cap region. InceptionV3's heatmaps showed slightly more diffuse activation across broader image regions, which may reflect its Inception module architecture that processes features at multiple scales simultaneously — resulting in less spatially precise but still biologically relevant attention patterns.

---

## F. Model Comparison & Improvement

**14. Which model would you recommend for deployment? Why?**

Answer: **MobileNetV2** is the recommended model for deployment based on four key factors. First, it achieved the highest validation accuracy (**97.70%**), best F1-score (**97.88%**), and highest AUC (**0.9989**) among all six models tested across all lab works. Second, MobileNetV2 is specifically designed for mobile and edge deployment — its depthwise separable convolution architecture produces a significantly smaller model file compared to EfficientNetB0 and InceptionV3, making it ideal for on-device inference in a mobile mushroom identification app. Third, it achieved the fastest inference speed, which translates to lower latency in real-time prediction scenarios. Fourth, when converted to TensorFlow Lite format, MobileNetV2 maintains strong accuracy while operating efficiently on mobile CPUs without requiring a GPU — critical for field use by hikers and foragers in remote locations without internet connectivity.

**15. How can you further improve your best-performing model?**

Answer: MobileNetV2 can be further improved through several strategies. First, **fine-tuning the top layers** — unfreezing the last 30–50 layers of the MobileNetV2 base and retraining at a very low learning rate (1e-5) would allow the pretrained weights to adapt more precisely to mushroom-specific features rather than relying solely on ImageNet representations. Second, **expanding the dataset** — collecting photos of multiple distinct specimens per species (not just rotational angles of one mushroom) would dramatically improve real-world generalization. Third, **test-time augmentation (TTA)** — averaging predictions across multiple augmented versions of the same test image would reduce prediction variance and improve confidence scores. Fourth, **ensemble methods** — combining predictions from MobileNetV2 and EfficientNetB0 through probability averaging would push accuracy beyond what either model achieves individually, since both achieved nearly identical performance at 97.70% and 97.30% respectively.

---

## G. Real-World Application

**16. How can your model be applied in real-world scenarios?**

Answer: The trained MobileNetV2 model has several practical real-world applications. The most direct is a **mushroom identification and safety app** for hikers, foragers, and nature enthusiasts — users photograph a wild mushroom and receive instant species identification along with safety information, flagging toxic species like *Amanita muscaria*, *Omphalotus olearius*, *Conocybe apala*, and *Agaricus xanthodermus* while confirming safe edible species like *Cantharellus lateritius* and *Hericium erinaceus*. Beyond consumer apps, the model could support **mycology research** by automating species logging in field surveys, assist **poison control centers** by providing rapid preliminary identification from patient-submitted photos, and contribute to **biodiversity monitoring** by enabling citizen scientists to document mushroom sightings with accurate species labels across geographic regions.

**17. What are the risks of deploying an inaccurate model?**

Answer: Deploying an inaccurate mushroom classification model carries serious risks across multiple dimensions. The most critical is **public safety risk** — a false negative that fails to identify a toxic species like *Amanita muscaria* or *Conocybe apala* as dangerous could directly lead to mushroom poisoning, which can cause liver failure, neurological damage, or death. The Teachable Machine model's 24.40% accuracy demonstrated how a model with misaligned labels could confidently produce wrong predictions — a catastrophic outcome in a safety-critical application. Beyond safety, there are **legal and ethical risks** — deploying a model that causes harm through incorrect predictions could expose developers to liability. There are also **ecological risks** — incorrect species identification in biodiversity monitoring would corrupt scientific datasets and lead to flawed conservation decisions. These risks underscore why thorough evaluation using multiple metrics (not just accuracy) and rigorous testing before deployment are non-negotiable requirements for any real-world AI system.

**18. How can this system be integrated into a mobile/web app?**

Answer: The MobileNetV2 model can be integrated into both mobile and web applications through multiple deployment pathways. For **mobile deployment**, the saved `.keras` model is converted to **TensorFlow Lite (TFLite)** format using `tf.lite.TFLiteConverter`, reducing the model size for on-device inference on Android and iOS without requiring an internet connection — essential for field use in remote foraging locations. The mobile app would capture an image, resize it to 224×224, normalize pixel values to [0,1], run inference through the TFLite interpreter, and display the top-3 predicted species with confidence scores and toxicity warnings. For **web deployment**, the model is served via a **Flask or FastAPI REST API** where users upload photos through a browser interface and receive JSON responses containing species predictions. For **scalable cloud deployment**, the API can be containerized with Docker and hosted on **Google Cloud Run** or **AWS Lambda** with automatic scaling to handle variable user loads. Critical production features should include a confidence threshold (reject predictions below 80%), geographic species filtering, multi-image averaging for improved accuracy, and a mandatory disclaimer that automated identification should not replace expert mycological verification.

