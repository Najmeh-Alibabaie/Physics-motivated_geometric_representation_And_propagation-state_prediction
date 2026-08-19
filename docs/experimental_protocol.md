Experimental Protocol and Evaluation Implementation

This document describes the implementation of the experiments reported in the paper. It specifies the data preparation, train/test partitioning, feature representations, machine-learning configuration, evaluation metrics, analytical baseline, and result aggregation procedures used by the released implementation.

1. Overview

The evaluation pipeline compares multiple geometric representations for propagation-state classification using a common machine-learning classifier and a common data-partitioning protocol. The propagation states considered in the experiments are Blocked, LOS (Line-of-Sight), and NLOS (Non-Line-of-Sight).

The implementation is organized around two evaluation procedures:

Representation comparison, in which different geometric feature sets are evaluated using the same XGBoost classifier.
Analytical-model and feature-based comparison, in which learned geometric representations are compared with distance-based representations and the standardized 3GPP TR 38.901 UMa LOS probability model.

The implementation uses transmitter-level grouping when constructing train/test partitions to prevent links originating from the same transmitter from appearing in both partitions.

2. Data Preparation

For each scenario, the environment-derived geometric features and communication-link information are stored in separate CSV files.

The geometric CSV file contains the descriptors extracted from the corresponding 3D environmental representation. The communication CSV file contains the transmitter (TX), receiver (RX), and propagation-state label (Status). The two files are joined using the (TX, RX) pair:

df = pd.merge(
    geo_df,
    link_df[["TX", "RX", "Status", "Received_Power_dBm"]],
    on=["TX", "RX"],
    how="inner"
)


Only links for which both geometric descriptors and communication information are available are retained. The propagation-state label is taken from the Status column.

The geometric descriptors used by a particular experiment are selected through an explicit feature list. Consequently, the classifier receives only the descriptors associated with the representation being evaluated.

For the representation-comparison experiment, three feature sets are evaluated independently:

features1: proposed geometric representation;
features2: baseline representation corresponding to Hussein et al.;
features3: baseline representation corresponding to Ladan et al.

All three representations are evaluated using the same communication links and the same classifier configuration.

3. Transmitter-Level Train/Test Partitioning

To avoid information leakage between training and testing data, samples are grouped according to their transmitter identifier:

groups = df["TX"]


The implementation uses GroupShuffleSplit with a test-group fraction of 20%:

splitter = GroupShuffleSplit(
    n_splits=10,
    test_size=0.2,
    random_state=42
)


The released representation-comparison implementation obtains one train/test partition from the splitter:

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)


Thus, all links associated with a given transmitter are assigned exclusively to either the training or testing subset.

This procedure is important because multiple links generated from the same transmitter can exhibit highly correlated geometric characteristics. Randomly splitting individual links could therefore result in highly similar samples occurring in both training and testing data and would lead to an overly optimistic estimate of generalization performance.

The random seed is fixed to 42 to make the partition deterministic.

4. Class-Balanced Training

The propagation-state classes are not assumed to be equally represented. To reduce the influence of class imbalance during model training, sample weights are computed using scikit-learn's balanced class-weight scheme:

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


The resulting weights are supplied directly to XGBoost during model fitting:

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


No synthetic oversampling or undersampling is performed.

5. XGBoost Configuration

All learned representations use the same XGBoost classifier configuration. The purpose is to isolate the effect of the input representation rather than optimize the classifier independently for each feature set.

The principal parameters are:

Parameter	Value
n_estimators	800
max_depth	10
learning_rate	0.05
subsample	0.9
colsample_bytree	0.9
objective	multi:softmax
num_class	3
random_state	42
n_jobs	-1

The implementation does not explicitly specify the remaining XGBoost regularization parameters. Therefore, their values are inherited from the installed XGBoost version's defaults.

The classifier directly predicts one of the three propagation-state classes.

6. Representation Comparison

The function evaluate_feature_set() performs the main representation-comparison experiment.

For each scenario and representation, the function:

loads the geometric and communication CSV files;
joins them using (TX, RX);
selects the specified feature columns;
obtains the propagation-state labels from Status;
groups samples by transmitter;
creates the transmitter-level train/test partition;
computes balanced training weights;
trains the XGBoost classifier;
predicts the propagation state of the test samples;
computes class-wise and aggregate performance metrics; and
stores the results for later aggregation.

The relevant implementation pattern is:

model, results = evaluate_feature_set(
    representation_name,
    scenario_name,
    geometry_file,
    link_file,
    feature_set,
    results
)


The experiment is repeated for each scenario and for each of the three representations.

7. Evaluation Metrics

The implementation reports precision, recall, and F1-score separately for the three propagation states:

Blocked;
LOS;
NLOS.

It additionally reports:

overall accuracy;
macro precision;
macro recall;
macro F1-score.

The class-specific metrics are obtained from classification_report(). Macro-averaged metrics assign equal importance to each propagation-state class, making them useful when class frequencies are imbalanced.

The results are accumulated in a tabular structure and exported as a CSV file:

results = pd.DataFrame(results)
results.to_csv(OUTPUT_DIR1, index=False)


The resulting file contains one row for each evaluated representation/scenario combination.

8. Analytical 3GPP TR 38.901 Baseline

The standardized baseline is implemented independently from the learned feature representations.

The function model_3GPP_38901() implements the Urban Macro (UMa) LOS probability relationship from 3GPP TR 38.901:

PLOS = (
    np.minimum(18.0 / d, 1.0)
    * (1.0 - np.exp(-d / 63.0))
    + np.exp(-d / 63.0)
)


where d is the three-dimensional transmitter-receiver distance.

The probability is converted into a deterministic propagation-state prediction using a threshold of 0.5:

pred = np.where(PLOS > 0.5, 1, 2)


Accordingly:

1 represents LOS;
2 represents NLOS.

Because the standardized model does not provide a separate Blocked state, Blocked links are treated as NLOS for this comparison.

The analytical model does not undergo machine-learning training. It directly produces predictions from the three-dimensional link distance.

9. Repeated Evaluation of Learned Representations

The second evaluation implementation evaluates each learned feature representation using repeated transmitter-level train/test partitions.

For each split, the following operations are performed independently:

construct the training and testing sets;
calculate class-balanced sample weights from the training labels;
train a new XGBoost model;
predict the test labels;
calculate accuracy, macro precision, macro recall, and macro F1-score.

The metrics from the individual partitions are then averaged:

return pd.DataFrame(results).mean()


This produces a mean performance estimate over the generated transmitter-level partitions.

10. Scenario-Level Result Aggregation

For every scenario, the implementation loads the corresponding geometric and communication files and evaluates all feature sets:

for name, features in feature_sets.items():
    metrics = evaluate_feature_set(df, features)


The analytical 3GPP baseline is then evaluated on the same scenario:

metrics = evaluate_3gpp(df)


The results are stored together using a common schema containing:

Scenario;
Model;
Accuracy;
Precision;
Recall;
F1.

This allows the learned geometric representations and the analytical baseline to be aggregated using the same evaluation metrics.

The final summary is obtained by averaging the scenario-level results for each model:

summary = (
    results.groupby("Model")[["Accuracy", "Precision", "Recall", "F1"]]
    .mean()
    .mul(100)
    .round(2)
)


The resulting summary is exported to:

Summaries.to_csv(OUTPUT_DIR2, index=False)


The reported percentages are therefore obtained by multiplying the mean metric values by 100 and rounding them to two decimal places.

11. Reproducibility

The implementation fixes the random seed of the data partitioning and XGBoost classifier to 42. The same classifier configuration is used for the learned feature representations.

To reproduce the reported experiments, users should execute the notebooks in the order specified by the repository and ensure that the required input CSV files and feature definitions are available at the paths specified by the scenario configuration.

The released implementation is intended to reproduce the experimental procedure and reported results rather than provide a general-purpose propagation simulator. Ray-tracing outputs are not required during model inference; they are used to provide the propagation-state labels used during supervised evaluation.

12. Relationship to the Paper

The implementation described here corresponds primarily to the following parts of the paper:

Performance Evaluation — representation comparison;
Experimental Setup — classifier configuration, class balancing, and transmitter-level partitioning;
Comparison With Existing Approaches — geometry-based feature representations;
Comparison With Analytical and Standardized Propagation Models — distance-based and 3GPP TR 38.901 comparisons;
Cross-Environment Generalization Analysis — transmitter/group-based evaluation procedures where applicable;
Ablation Study — repeated evaluation of individual and combined descriptor families.

The repository documentation should be considered the implementation-level companion to the methodological description in the paper.
