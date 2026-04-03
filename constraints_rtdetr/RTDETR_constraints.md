# 2.3.2 Design Alternative: RT-DETR

The RT-DETR architecture is a transformer-based real-time object detection
network designed to localize and classify solder-defect regions directly from
PCB images. In this project, the model was used to detect four solder-related
classes from YOLO-format bounding-box annotations collected from macro PCB
images. RT-DETR was selected because it combines end-to-end detection with
strong multi-object localization, which is suitable for solder inspection tasks
where the system must identify both the location and type of defect in a single
forward pass. However, this design also introduces practical constraints
related to dataset scale, class imbalance, inference latency, training
convergence, and training-data dependency.

## A. Engineering Principles of Alternative

The RT-DETR design integrates the engineering principles of end-to-end object
detection, transformer-based feature reasoning, and direct set prediction.
Instead of segmenting every pixel, the network predicts a set of object boxes
and class labels for visible solder regions in the image. This is important for
solder inspection because defects such as excess solder, insufficient solder,
and solder spike typically occupy compact regions, so the system must localize
each defect candidate and assign the correct class efficiently.

The model also applies the principle of global feature interaction through
transformer decoding. This allows the detector to relate local solder
appearance to the larger visual context of nearby pads, traces, and
components. As a result, RT-DETR can detect multiple solder conditions in a
single image while maintaining stronger deployment speed than a dense
segmentation design. However, the effectiveness of this principle remains
dependent on the quantity, balance, and representativeness of the available
training annotations.

## B. Architecture of Design Alternative

The implemented model uses the pretrained `rtdetr-l` architecture fine-tuned on the solder-defect dataset stored in YOLO detection format. Input images are trained at `640 x 640` resolution with bounding-box supervision for four defect classes: Good Solder (`good`), Excess Solder (`exc_solder`), Insufficient Solder (`poor_solder`), and Solder Spike (`spike`). Training was performed with batch size `2`, early stopping patience `30`, and automatic mixed precision on CUDA using the RTX 3050 Laptop GPU available in the workspace.

The best deployed checkpoint is the current `best.pt` under `rtdetr/runs/solder_defects_rtdetr/weights/`. Validation of this checkpoint produced `92.64%` precision, `95.94%` recall, `97.50%` mAP50, and `87.81%` mAP50-95 on the validation set. The complete training history shows that the run stopped after `69` epochs, with the strongest mAP50 at epoch `68` and the strongest mAP50-95 at epoch `69`.

## C. Constraints

In this section, the constraints establish the key dataset, performance, efficiency, and training limitations of the RT-DETR design. These constraints define how accurately the detector identifies solder-defect classes, how efficiently it can process PCB images, and how practical it is for deployment and future improvement using the currently available training annotations.

### Quick Result Explanations

The following condensed results can be used when a short explanation is needed in the documentation or manuscript:

- `312 train / 217 val, 4 classes`: This means the RT-DETR model was trained using 312 labeled images and validated using 217 separate images. The detection task includes four output classes: good solder, excess solder, insufficient solder, and solder spike.
- `97.50% mAP50 and 87.81% mAP50-95`: These are the current best-checkpoint validation metrics for object detection quality. The model is strong at coarse localization and remains solid under stricter IoU thresholds.
- `418.45 ms CPU wall-clock latency`: This is the measured mean CPU inference latency over 50 validation images using the current best checkpoint. It is usable for offline or semi-automatic inspection, but still slow for strict real-time CPU deployment.
- `69 epochs with best mAP50 at epoch 68 and best mAP50-95 at epoch 69`: This means the training run converged late and did not need the originally requested 100 epochs. The best high-IoU detection quality was reached at the end of the run.
- `high data dependency`: This describes the model's reliance on balanced defect annotations. The smallest class, insufficient solder, has only 91 labeled instances across train and validation, which limits class robustness.

The saved plots that support these points are:

- Dataset split and class composition: `constraints_rtdetr/figures/rtdetr_dataset_distribution.png`
- Validation performance summary: `constraints_rtdetr/figures/rtdetr_performance_summary.png`
- Training behavior across epochs: `constraints_rtdetr/figures/rtdetr_training_curves.png`
- CPU latency summary: `constraints_rtdetr/figures/rtdetr_latency_summary.png`
- Data dependency: `constraints_rtdetr/figures/rtdetr_data_dependency.png`
- Validation artifacts reused from the detector evaluation: `constraints_rtdetr/figures/rtdetr_confusion_matrix_normalized.png`, `constraints_rtdetr/figures/rtdetr_BoxPR_curve.png`, and `constraints_rtdetr/figures/rtdetr_BoxF1_curve.png`

The notebook version of this section reuses `constraints_rtdetr/summary.json` and the saved figures in `constraints_rtdetr/figures` directly, so the documentation can be refreshed without touching the main RT-DETR app or training code.

### Dataset

This constraint is based on the size, composition, and class balance of the labeled dataset used to train and validate the RT-DETR model.

| Metric | Value |
| --- | --- |
| Total Labeled Images | 529 |
| Training Images | 312 |
| Validation Images | 217 |
| Input Resolution | 640 x 640 |
| Number of Classes | 4 |
| Classes | good solder, excess solder, insufficient solder, solder spike |
| Smallest Class by Instance Count | Insufficient Solder (`poor_solder`): 91 instances |
| Largest-to-Smallest Class Ratio | 3.46x |

The dataset defines the learning capacity of the detector. Based on the saved summary, the class distribution is imbalanced: Excess Solder (`exc_solder`) has 315 labeled instances, Good Solder (`good`) has 311, Solder Spike (`spike`) has 231, while Insufficient Solder (`poor_solder`) has only 91. That gap is large enough to influence class-level robustness, especially for minority or visually ambiguous defect categories.

### Performance (Accuracy)

This constraint is based on the model's ability to correctly detect solder-defect classes on the validation set. Precision and recall describe detection reliability, while mAP50 and mAP50-95 reflect localization quality across IoU thresholds.

| Metric | Value |
| --- | --- |
| Validation Precision | 92.64% |
| Validation Recall | 95.94% |
| Validation mAP50 | 97.50% |
| Validation mAP50-95 | 87.81% |
| Weakest Per-Class mAP50-95 | 80.96% for solder spike |
| Minority-Class mAP50-95 | 82.48% for insufficient solder |

These metrics show that the detector is strong overall, especially at the
standard mAP50 threshold. However, stricter localization quality and
class-level robustness still vary across defect categories. The strongest
per-class mAP50-95 values belong to Good Solder and Excess Solder, both above
`93%`, while Insufficient Solder and Solder Spike remain notably lower. Therefore, the stronger
constraint is not whether the model detects obvious solder regions, but whether
it can maintain the same quality across all defect types.

### Efficiency (Inference Latency)

This constraint is based on the average time required for the model to process a single image during inference. The latency benchmark was measured on CPU over 50 validation images using the current best RT-DETR checkpoint.

| Metric | Value |
| --- | --- |
| Mean Preprocess Latency | 2.48 ms/image |
| Mean Forward Inference Latency | 409.88 ms/image |
| Forward Inference Standard Deviation | 13.02 ms |
| Mean Postprocess Latency | 0.64 ms/image |
| Mean Wall-Clock Latency | 418.45 ms/image |
| Wall-Clock Standard Deviation | 40.29 ms |

This metric evaluates the responsiveness of the detector when deployed outside the training environment. The measured CPU inference time indicates that RT-DETR is practical for manual or semi-automatic inspection, but it is still too slow for strict real-time CPU-only deployment without additional optimization, lighter models, reduced image size, or stronger hardware.

### Training Epochs (Validation Accuracy)

This constraint is based on how the model behaved across training epochs, particularly the point at which validation quality stabilized.

| Metric | Value |
| --- | --- |
| Total Training Epochs Completed | 69 |
| Final Validation mAP50 | 97.50% at epoch 69 |
| Final Validation mAP50-95 | 87.82% at epoch 69 |
| Highest Validation mAP50 | 98.75% at epoch 68 |
| Best Validation mAP50-95 | 87.82% at epoch 69 |
| Total Training Time | 4542.51 s (75.71 min) |

These values show that the detector improved steadily over a long training run and reached its best strict-IoU performance at the end of training. The difference between epoch 68 and epoch 69 is small, but it confirms that the model was still extracting useful improvements near the end of the run. In practice, mAP50-95 is the better criterion for final checkpoint selection because it reflects stronger localization quality than mAP50 alone.

### Data Dependency (Training Data Efficiency)

This constraint is based on how strongly the detector depends on the size and balance of the available labeled training data.

| Metric | Value |
| --- | --- |
| Training Data Efficiency | Not directly measurable from a single run |
| Smallest Class Instance Count | 91 insufficient solder instances |
| Largest-to-Smallest Class Ratio | 3.46x |
| Weakest Per-Class mAP50-95 | 80.96% for solder spike |
| Minority-Class mAP50-95 | 82.48% for insufficient solder |

This metric cannot be reported as a precise data-efficiency score because the
project did not include a controlled subset study using reduced fractions of
the training set. However, the saved dataset summary and per-class metrics
already show strong data dependency. The minority class Insufficient Solder
(`poor_solder`) has the fewest labeled instances, and both Insufficient Solder
and Solder Spike trail the
dominant classes in per-class mAP50-95. This indicates that defect robustness
is influenced by both annotation quantity and visual difficulty. As a result,
additional labeled examples for minority and ambiguous classes would likely
improve class balance more effectively than simply extending training duration.

## D. Evaluation Results

This section presents the evaluation results of the RT-DETR architecture based on the saved validation artifacts and the current best checkpoint validation summary. The evaluation focuses on aggregate detection quality, class-wise robustness, and training convergence across epochs.

The validation summary shows that RT-DETR is effective for solder-defect detection on the current dataset. Aggregate performance is strong, with `92.64%` precision, `95.94%` recall, `97.50%` mAP50, and `87.81%` mAP50-95. These values indicate that the model usually identifies the correct solder-defect regions and maintains good localization quality under stricter overlap thresholds.

The class-wise results provide a more realistic picture of the remaining limitations. `Good Solder` and `Excess Solder` both exceed `93%` mAP50-95, while `Insufficient Solder` falls to `82.48%` and `Solder Spike` to `80.96%`. This confirms that the weaker classes are not the dominant ones, and that class balance plus visual ambiguity still constrain performance.

The training-results figure shows that validation mAP50 and mAP50-95 increased
substantially across the 69 completed epochs while the training losses
decreased without major instability. The highest mAP50 occurred at epoch 68,
while the best mAP50-95 occurred at epoch 69. Overall, these results indicate
that RT-DETR is an effective detection-based design alternative for
solder-defect localization, but its class-level robustness remains limited by
class imbalance, minority-class coverage, and deployment latency on CPU.