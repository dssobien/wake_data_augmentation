#!python3

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import matthews_corrcoef


# def calc_f1(df):
#     df['recall'] = df['tp'] / (df['tp'] + df['fn'])
#     df['precision'] = df['tp'] / (df['tp'] + df['fp'])
#     df['f1'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
#     return df


def get_gt_arr(df):
    df_gt = df[df['model'] == 'Truth']
    iterations = sorted(list(df_gt["iteration"].unique()))
    rows = []
    for i in iterations:
        rows.append(df_gt[df_gt["iteration"] == i].iloc[0].name)
    df_gt = df_gt.loc[rows]
    gt_arr = np.where(np.array([np.array(x) for x in df_gt['conf']]) == 'True', 1, 0)
    assert (len(gt_arr) == len(iterations))
    return gt_arr


def pr_auc(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)


def f1_from_pr(y_true, y_pred):
    p, r, t = precision_recall_curve(y_true, y_pred)
    denom = np.where((p + r) == 0, 1e-10, (p + r))
    f1_arr = 2 * p * r / (denom)
    f1_max = f1_arr.max()
    t_max = t[f1_arr.argmax()]
    return f1_max, t_max


def calc_f1_max(row, gt_arr):
    hp_idx = row["iteration"]
    band = row["band"]
    y_true = gt_arr[hp_idx]
    y_pred = np.array(row["conf"], dtype="float")
    f1_max, t_max = f1_from_pr(y_true, y_pred)
    return [f1_max, round(t_max, 3)]


def calc_pr_auc(row, gt_arr, threshold):
    hp_idx = row["iteration"]
    band = row["band"]
    y_true = gt_arr[hp_idx]
    y_pred = np.array(row["conf"], dtype="float")
    return pr_auc(y_true, y_pred)


def calc_roc_auc(row, gt_arr, threshold):
    hp_idx = row["iteration"]
    band = row["band"]
    y_true = gt_arr[hp_idx]
    y_pred = np.array(row["conf"], dtype="float")
    return roc_auc_score(y_true, y_pred)


def calc_f1(row, gt_arr, threshold):
    hp_idx = row["iteration"]
    band = row["band"]
    y_true = gt_arr[hp_idx]
    y_pred = np.array(row["conf"], dtype="float")
    y_pred = np.where(y_pred > threshold, 1, 0)
    return f1_score(y_true, y_pred)


def calc_mcc(row, gt_arr, threshold):
    hp_idx = row["iteration"]
    band = row["band"]
    y_true = gt_arr[hp_idx]
    y_pred = np.array(row["conf"], dtype="float")
    y_pred = np.where(y_pred > threshold, 1, 0)
    return matthews_corrcoef(y_true, y_pred)


def output_metrics(df_input, metrics=['pr_auc'], threshold=0.6):
    # takes dataframe input and returns a dataframe with F1 and PR AUC
    # performance metrics calculated
    df = df_input.copy()
    gt_arr = get_gt_arr(df)
    df = df[df['model'] != 'Truth']
    for metric in metrics:
        metric_func = metrics_dict[metric]
        df[metric] = df.apply(lambda row: metric_func(row, gt_arr, threshold), axis=1)
    return df


def output_metrics_folds(file_path, fold=None, label=None, metrics=['pr_auc'], threshold=0.6):
    df = pd.read_csv(file_path,
            converters={"conf": lambda x: x.strip("[]").replace("'", "").split(", ")})
    final_results = df[df["iteration"] == -1]
    df = df[df["iteration"] != -1]
    df = output_metrics(df, metrics=metrics, threshold=threshold).fillna(0)
    # df = df[["iteration", "band", "model", "test_dir", "pr_auc", "conf"]]
    if fold is not None:
        df.insert(len(df.columns), "fold", fold)
    if label is not None:
        df.insert(len(df.columns), "label", label)
    for col in ['tp', 'tn', 'fp', 'fn']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df


def output_metrics_for_augmentation(path_name):
    df_all = None
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(4):
        file_names = glob.glob(f"{path_name}{i}/*.csv")
        file_name = [x for x in file_names if f"fold{i}" in x][0]
        # df = pr_auc_df(file_name, fold=i)
        for t in thresholds:
            df = output_metrics_folds(file_name, fold=i, label=f"threshold_{t}",
                                      metrics=['mcc', 'pr_auc'], threshold=t)
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])
    return df_all.reset_index(drop=True)


metrics_dict = {'pr_auc': calc_pr_auc, 'roc_auc': calc_roc_auc, 'f1': calc_f1, 'mcc': calc_mcc}
