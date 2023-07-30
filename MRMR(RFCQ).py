import functools
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, KFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import VarianceThreshold
from collections import Counter
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as imbpipeline
from mrmr import mrmr_classif
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import f_classif
from .main import mrmr_base


def parallel_df(func, df, series, n_jobs):
    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def _f_classif(X, y):
    def _f_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y):
    def _f_regression_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def f_classif(X, y, n_jobs):
    return parallel_df(_f_classif, X, y, n_jobs=n_jobs)


def f_regression(X, y, n_jobs):
    return parallel_df(_f_regression, X, y, n_jobs=n_jobs)


def _ks_classif(X, y):
    def _ks_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        x = x[x_not_na]
        y = y[x_not_na]
        return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()

    return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)


def ks_classif(X, y, n_jobs):
    return parallel_df(_ks_classif, X, y, n_jobs=n_jobs)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X, n_jobs):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)

    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs)


def encode_df(X, y, cat_features, cat_encoding):
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X


def rfcq_classif(
        X, y, K,
        relevance='rf', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a classification task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "ks" (kolmogorov-smirnov), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    if relevance == "f":
        relevance_func = functools.partial(f_classif, n_jobs=n_jobs)
    elif relevance == "ks":
        relevance_func = functools.partial(ks_classif, n_jobs=n_jobs)
    elif relevance == "rf":
        relevance_func = random_forest_classif
    else:
        relevance_func = relevance

    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, multi_class='ovr', average=average)

from sklearn.metrics import roc_auc_score

def class_vs_rest_roc_auc_score(y_test, y_pred):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    y_pred_bin = lb.transform(y_pred)
    roc_auc_scores = {}

    for i, class_label in enumerate(lb.classes_):
        class_indices = lb.transform([class_label])[0]
        rest_indices = ~class_indices
        y_test_class = y_test_bin[:, i]
        y_test_rest = y_test_bin[:, rest_indices]
        y_pred_class = y_pred_bin[:, i]
        y_pred_rest = y_pred_bin[:, rest_indices]
        roc_auc_scores[class_label] = roc_auc_score(y_test_class, y_pred_class, multi_class='ovr', average="macro")

    return roc_auc_scores

def calculate_specificity(cm):
    num_classes = cm.shape[0]
    specificity = {}

    for i in range(num_classes):
        tp = cm[i, i]
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        specificity[i] = 100*(tn / (tn + fp))

    return specificity

def calculate_sensitivity(cm):
    num_classes = cm.shape[0]
    sensitivity = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp

        sensitivity[i] = 100*(tp / (tp + fn))

    return sensitivity

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

df = pd.read_excel('all_patients.xlsx')
#df = df.set_index(['subjid'])
df = df.drop('number_annotations', axis = 1)
df = df[df.shape_VoxelVolume > 125]
df.drop(df[df['labels'] == 3].index, inplace = True) ## dropping label of the call OTHERS
df = df.set_index(['subjid', 'label'])
confidence = 0.95
seed = 0
df

# we select X and y
X = df.iloc[:, :-4] #leaving the sex and age features out for now
y = df.iloc[:, -1:]

pipelines = {
    'rf': imbpipeline([('smoteen', SMOTEENN(random_state=0)),
                       ('clf', RandomForestClassifier(random_state=0, class_weight="balanced"))]),
    'svc': imbpipeline([('smoteen', SMOTEENN(random_state=0)),
                        ('clf', CalibratedClassifierCV(LinearSVC(multi_class='ovr', random_state=0, class_weight="balanced")))])
    }

param_grids = {
    'rf': {'clf__n_estimators': [100, 250, 1000, 3500]},
    'svc': {'clf__base_estimator__C': [0.1, 1,10], 'clf__base_estimator__max_iter': [10000, 20000]}
}

cv_outer = RepeatedStratifiedKFold(n_splits=6, n_repeats=6, random_state=seed)

# Store the scores and metrics
scores_rf = {'acc': [], 'SD_acc': [], 'CI_acc': [], 'auc_tot': [], 'SD_auc_tot': [], 'CI_auc_tot': [], 'auc_class': [], 'SD_auc_class': [], 'CI_auc_class': [], 'Sen_mean': [], 'Sen_CI': [], 'Sen_SD': [], 'Spe_mean': [], 'Spe_CI': [], 'Spe_sd': []}
scores_svc = {'acc': [], 'SD_acc': [], 'CI_acc': [], 'auc_tot': [], 'SD_auc_tot': [], 'CI_auc_tot': [], 'auc_class': [], 'SD_auc_class': [], 'CI_auc_class': [], 'Sen_mean': [], 'Sen_CI':[], 'Sen_SD': [], 'Spe_mean': [], 'Spe_CI': [], 'Spe_sd': []}
best_features_RF = []
for name, pipeline in pipelines.items():
    # Initialize lists to store scores for each fold
    fold_acc = []
    fold_auc_tot = []
    fold_auc_class = []
    sensi = []
    speci = []
    features = []
    p_value = []
    feat = []
    value = []
    for train_index, test_index in cv_outer.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_normed = scaler.fit_transform(X_train)
        X_test_normed = scaler.transform(X_test)

        X_train_scal = pd.DataFrame(X_train_normed, columns=X_train.columns, index=X_train.index)
        X_test_scal = pd.DataFrame(X_test_normed, columns=X_test.columns, index=X_test.index)

        #select the best 15 features
        selected_feature_names = rfcq_classif(X_train_scal, y_train, K=5)
        X_train_new = X_train_scal[selected_feature_names]
        X_test_new = X_test_scal[selected_feature_names]
