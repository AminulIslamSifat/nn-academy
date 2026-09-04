---
title: "Evaluation Metrics"
slug: "034-evaluation-metrics"
description: "Accuracy lies. Learn precision, recall, F1, confusion matrices, and ROC/AUC from scratch in NumPy to truly understand your model."
track: "nn-beginner"
order: 7
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["032-mnist-from-scratch"]
---

# Evaluation Metrics

==Accuracy is a liar.== If 95% of your data is class A, a model that always predicts A gets 95% accuracy and is completely useless. You need metrics that reveal what your model actually does.

## Confusion Matrix

The foundation of everything else. For binary classification:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

```python
import numpy as np

def confusion_matrix_binary(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return np.array([[tp, fn], [fp, tn]])
```

<PyRunner
  cellId="034-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)
y_true = np.array([1,1,1,1,1,0,0,0,0,0])
y_pred = np.array([1,1,1,0,0,1,0,0,0,0])

tp = np.sum((y_true==1)&(y_pred==1))
fn = np.sum((y_true==1)&(y_pred==0))
fp = np.sum((y_true==0)&(y_pred==1))
tn = np.sum((y_true==0)&(y_pred==0))

print("Confusion Matrix:")
print(f"              Pred+   Pred-")
print(f"  Actual+  [ {tp:3d}     {fn:3d} ]")
print(f"  Actual-  [ {fp:3d}     {tn:3d} ]")
print(f"\nAccuracy: {(tp+tn)/(tp+fn+fp+tn):.2f}")
print(f"But look: the model missed 2 positives AND falsely flagged 1 negative")
`}
/>

## Precision, Recall, F1

Three numbers that tell the real story:

<BlockMath latex="\text{Precision} = \frac{TP}{TP + FP}" />
<BlockMath latex="\text{Recall} = \frac{TP}{TP + FN}" />
<BlockMath latex="F_1 = \frac{2 \cdot P \cdot R}{P + R}" />

- **Precision**: Of everything predicted positive, how much actually is? ("When you say yes, are you right?")
- **Recall**: Of all actual positives, how many did you catch? ("Did you find them all?")
- **F1**: Harmonic mean balancing both.

```python
def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
```

<PyRunner
  cellId="034-cell-2"
  defaultCode={`import numpy as np

# Scenario: Medical screening (catching disease)
# High recall matters more than high precision
tp, fp, fn, tn = 90, 30, 10, 870

p = tp/(tp+fp)
r = tp/(tp+fn)
f1 = 2*p*r/(p+r)
acc = (tp+tn)/(tp+fp+fn+tn)

print("Medical Screening Example:")
print(f"  Accuracy:  {acc:.3f}")
print(f"  Precision: {p:.3f}  (of flagged patients, {p*100:.0f}% actually sick)")
print(f"  Recall:    {r:.3f}  (caught {r*100:.0f}% of sick patients)")
print(f"  F1:        {f1:.3f}")
print(f"\n⚠️ 96% accuracy sounds great, but 10 sick patients were MISSED")
print(f"   In medical context, recall matters more than precision")
`}
/>

> [!IMPORTANT] When to Optimize What
> - **High precision needed**: Spam detection (don't delete legitimate emails)
> - **High recall needed**: Disease screening (don't miss sick patients)
> - **Balanced (F1)**: Most general classification tasks

## Multi-Class Metrics

For MNIST (10 classes), compute per-class metrics then average:

```python
def multiclass_metrics(y_true, y_pred, num_classes=10):
    results = {}
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        results[c] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
    
    # Macro average (equal weight per class)
    macro_f1 = np.mean([results[c]["f1"] for c in range(num_classes)])
    return results, macro_f1
```

<PyRunner
  cellId="034-cell-3"
  defaultCode={`import numpy as np

np.random.seed(42)
# Simulate MNIST predictions with some classes harder than others
y_true = np.repeat(np.arange(10), 100)  # 100 samples per digit
y_pred = y_true.copy()

# Make some digits confused
confusions = [(3,8,15), (5,6,10), (4,9,8), (7,1,5)]
for true_cls, pred_cls, count in confusions:
    mask = (y_true == true_cls)
    indices = np.where(mask)[0][:count]
    y_pred[indices] = pred_cls

# Compute per-class metrics
print(f"{'Digit':>5} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7}")
print("─" * 42)
f1s = []
for c in range(10):
    tp = np.sum((y_true==c)&(y_pred==c))
    fp = np.sum((y_true!=c)&(y_pred==c))
    fn = np.sum((y_true==c)&(y_pred!=c))
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    f1s.append(f1)
    print(f"  {c:3d}  | {p:6.3f} | {r:6.3f} | {f1:6.3f} | {tp+fn:7d}")

print(f"\nMacro F1: {np.mean(f1s):.3f}")
print(f"Accuracy: {np.mean(y_true==y_pred):.3f}")
print(f"\n💡 Digits 3↔8, 5↔6, 4↔9 are commonly confused — even by humans!")
`}
/>

## ROC Curve & AUC

For binary classifiers with probability outputs. Plot True Positive Rate vs False Positive Rate at every threshold:

<BlockMath latex="\text{TPR} = \frac{TP}{TP+FN}, \quad \text{FPR} = \frac{FP}{FP+TN}" />

AUC (Area Under Curve) summarizes overall performance: 0.5 = random, 1.0 = perfect.

<PyRunner
  cellId="034-cell-4"
  defaultCode={`import numpy as np

np.random.seed(42)
# Simulated probabilities for binary classification
y_true = np.array([1]*50 + [0]*50)
y_scores = np.concatenate([
    np.clip(np.random.beta(3, 1.5, 50), 0, 1),  # positives skewed high
    np.clip(np.random.beta(1.5, 3, 50), 0, 1),  # negatives skewed low
])

# Compute ROC curve
thresholds = np.sort(np.unique(y_scores))[::-1]
tprs, fprs = [], []

for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    tn = np.sum((y_true==0)&(y_pred==0))
    tprs.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0)

tprs, fprs = np.array(tprs), np.array(fprs)

# AUC via trapezoidal rule
sorted_idx = np.argsort(fprs)
auc = np.trapz(tprs[sorted_idx], fprs[sorted_idx])

print(f"ROC-AUC: {auc:.3f}")
print(f"\nSample points on ROC curve:")
print(f"{'Threshold':>9} | {'TPR':>5} | {'FPR':>5}")
print("─" * 26)
for i in range(0, len(thresholds), max(1, len(thresholds)//8)):
    print(f"{thresholds[i]:9.3f} | {tprs[i]:5.3f} | {fprs[i]:5.3f}")

print(f"\n✅ AUC > 0.9 = excellent classifier")
print(f"   AUC ≈ 0.5 = random guessing")
`}
/>

> [!NOTE] When to Use AUC
> AUC is threshold-independent — it evaluates ranking quality, not a specific decision boundary. Use it when you care about relative ordering (e.g., risk scoring) rather than a fixed threshold.

## Metric Selection Guide

| Scenario | Primary Metric | Why |
|----------|---------------|-----|
| Balanced classes | Accuracy | Simple, interpretable |
| Imbalanced classes | F1 or AUC | Accuracy is misleading |
| Costly false positives | Precision | Minimize false alarms |
| Costly false negatives | Recall | Don't miss positives |
| Probability calibration | Brier Score | Measures probability quality |
| Ranking quality | AUC-ROC | Threshold-independent |

<Quiz
  chapterSlug="034-evaluation-metrics"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "A spam filter has 99% accuracy but catches only 60% of spam. What's the problem?",
      options: [
        "Accuracy is fine, 60% recall is acceptable",
        "The dataset is likely imbalanced (most emails aren't spam), so accuracy is misleading — recall of 60% means 40% of spam reaches the inbox",
        "Need more training data",
        "Should use MSE instead"
      ],
      correctIndex: 1,
      explanation: "If 95% of emails are legitimate, predicting 'not spam' for everything gives 95% accuracy. The 99% accuracy hides terrible spam detection. Always check precision/recall for imbalanced problems.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the F1 score when precision = 0.8 and recall = 0.5?",
      code: "p, r = 0.8, 0.5\nf1 = 2*p*r/(p+r)\nprint(f'{f1:.3f}')",
      options: ["0.650", "0.615", "0.571", "0.800"],
      correctIndex: 1,
      explanation: "F1 = 2 × 0.8 × 0.5 / (0.8 + 0.5) = 0.8 / 1.3 ≈ 0.615. The harmonic mean penalizes imbalance — F1 is closer to the lower value (recall).",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does AUC = 0.5 mean?",
      options: [
        "Perfect classifier",
        "The classifier is no better than random guessing",
        "50% accuracy",
        "The model predicts all positives"
      ],
      correctIndex: 1,
      explanation: "AUC = 0.5 means the ROC curve follows the diagonal line — the model's probability scores have no discriminative power. It ranks positives and negatives equally.",
      randomize: true,
    }
  ]}
/>
