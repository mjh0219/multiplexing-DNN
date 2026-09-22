# Improved Attention Comparison on UCI Covertype

This experiment uses 100,000 records from the UCI Covertype dataset:

- 60,000 training records
- 20,000 validation records
- 20,000 testing records
- 54 input features
- 7 output classes

The complete online dataset contains 581,012 records:

https://archive.ics.uci.edu/dataset/31/covertype

## Verified command

```bash
python3 -B attention_method_comparison.py \
  --dataset covertype \
  --max-samples 100000 \
  --epochs 25 \
  --finetune-epochs 8 \
  --hidden-dim 256 \
  --second-dim 128 \
  --token-dim 8 \
  --num-heads 2 \
  --keep-ratio 0.75 \
  --output-dir attention_comparison_covertype_100k_results
```

## Measured results

| Method | Total time (s) | NN space | Accuracy |
|---|---:|---:|---:|
| FC DNN | 50.4217 | 48,647 | 0.9069 |
| Feature Attention | 43.1371 | 48,701 | 0.9108 |
| Self-Attention | 314.5719 | 113,911 | 0.8917 |
| Proposed SANN | 91.4250 | 36,871 | 0.9204 |

The proposed SANN achieved 92.04% test accuracy. Compared with FC DNN, it
improved accuracy by 1.35 percentage points and reduced effective network
space by 24.21%. SANN retained 34,816 structural edges, and edge selection
took 0.0270 seconds.

The wider models use batch normalization and a 10% dropout rate. SANN retains
75% of candidate structural edges, providing a balance between predictive
accuracy and network-space reduction.
