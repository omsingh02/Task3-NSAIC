1. What was the single most impactful change you made?
Addressing the severe class imbalance (1906:334) and non-linear feature relationships by switching from LogisticRegression to RandomForestClassifier with balanced class weights.

2. Why does it work — mathematically or statistically?
The dataset has an ~85% negative class imbalance. Logistic Regression optimizes for overall accuracy, heavily favoring the majority class, resulting in poor minority class recall and a low F1-Score. Additionally, Logistic Regression assumes linear decision boundaries. RandomForest handles class imbalance intrinsically using recursive partitioning and `class_weight='balanced'` (which applies inverse frequency weights). It also inherently handles varying scales among features without needing explicit scaling.

3. Table: baseline metric | improved metric | delta
| Metric | Baseline | Improved | Delta |
| --- | --- | --- | --- |
| F1-Score | 0.1882 | 0.5189 | +175.68% |
