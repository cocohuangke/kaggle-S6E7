"""Reference notebook reproduction: ps-s6-ep6-realmlp-0-95090.ipynb

Exact reproduction of the reference notebook's RealMLP pipeline.
Adapted for local execution (paths, CUDA) with ZERO algorithmic changes.

Reference: LB=0.95090, OOF=0.95065
"""
import math
import random
import warnings
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
print("PyTorch version:", torch.__version__)

def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

SEED = 63
seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# Load Data
# ============================================================
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
print("Train shape:", train.shape)
print("Test shape :", test.shape)

# ============================================================
# Feature Engineering
# ============================================================
ID     = 'id'
TARGET = 'health_condition'
train[TARGET] = train[TARGET].map({'at-risk': 0, 'fit': 1, 'unhealthy': 2})
X        = train.drop([ID, TARGET], axis=1)
train_id = train[ID]
y        = train[TARGET]
X_test   = test.drop([ID], axis=1)
test_id  = test[ID]
del train, test
print("X      init shape:", X.shape)
print("X_test init shape:", X_test.shape, "\n")

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols), "\n")

category_map     = {}
important_combos = sorted([('heart_rate', 'bmi')])

EPS = 1e-6

def safe_div(a, b):
    return a / np.where(np.abs(b) < EPS, EPS, b)

def add_gp_features(df):
    sd  = df['sleep_duration'].fillna(0).values.astype(np.float64)
    bmi = df['bmi'].fillna(0).values.astype(np.float64)
    wi  = df['water_intake'].fillna(0).values.astype(np.float64)
    cal = df['calorie_expenditure'].fillna(0).values.astype(np.float64)
    sc  = df['step_count'].fillna(0).values.astype(np.float64)
    gp = {
        '_gp_sin_sleep':                    np.sin(sd),
        '_gp_sigmoid_sleep':                1.0 / (1.0 + np.exp(-np.clip(sd, -20, 20))),
        '_gp_log1p_bmi':                    np.log1p(np.abs(bmi)),
        '_gp_sin_sleep_x_bmi':              np.sin(sd) * bmi,
        '_gp_tanh_water':                   np.tanh(wi),
        '_gp_sin_sleep_bmi_sigmoid':        np.sin(sd) * (1.0 / (1.0 + np.exp(-np.clip(bmi, -20, 20)))),
        '_gp_tanh_water_sin_sleep_sig_cal': np.tanh(wi) * np.sin(sd) * (1.0 / (1.0 + np.exp(-np.clip(cal, -20, 20)))),
        '_gp_sin_sleep_log_sc_log_bmi':     np.sin(sd) * np.log1p(np.abs(sc)) - np.log1p(np.abs(bmi)),
        '_gp_sin_sleep_minus_tanh_cal':     np.sin(sd) - np.tanh(cal),
    }
    for k, v in gp.items():
        arr = np.array(v, dtype=np.float32)
        df[k] = np.where(np.isfinite(arr), arr, 0.0)
    return df

def feature_engineering(df, fit=False):
    for col in cat_cols:
        df[col] = df[col].fillna("missing")
    for col in num_cols:
        df[col] = df[col].fillna(0.0)

    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques  = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes    = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = codes
        df[col] = df[col].astype('category')

    for col in num_cols:
        cat_name = f"{col}_cat_"

        if col == 'calorie_expenditure':
            df[cat_name] = (df[col] // 5).astype('int32').astype('category')
            continue

        if col == 'water_intake':
            df['water_intake_cat2_'] = (df[col] * 50).round().astype('int32').astype('category')

        if col == 'sleep_duration':
            df['sleep_duration_cat2_'] = (df[col] * 10).round().astype('int32').astype('category')
            df['sleep_duration_cat2_'] = (df[col] * 10).round().astype('int32').astype('category')

        if col == 'bmi':
            df['bmi_cat2_'] = (df[col] - 18.5).round().astype('int32').astype('category')
            df['bmi_cat1_'] = (24.9 - df[col]).round().astype('int32').astype('category')

        if col == 'heart_rate':
            df['heart_rate_cat2_'] = (df[col] - 60).round().astype('int32').astype('category')
            df['heart_rate_cat2_'] = (100 - df[col]).round().astype('int32').astype('category')

        round_level = -1
        if fit:
            round_flag = (col == 'step_count')
            series     = df[col].round(round_level) if round_flag else df[col]
            codes, uniques = series.factorize()
            category_map[col] = {'uniques': uniques, 'round_flag': round_flag}
        else:
            round_flag = category_map[col]['round_flag']
            uniques    = category_map[col]['uniques']
            series     = df[col].round(round_level) if round_flag else df[col]
            code_map   = {cat: i for i, cat in enumerate(uniques)}
            codes      = series.map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')

    bin_config = {'sleep_duration': [70], 'water_intake': [10]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb     = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                          strategy='quantile', subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bin_name] = kb
            else:
                binned = category_map[bin_name].transform(df[[col]]).ravel().astype('int32')
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype('category')

    for cols in important_combos:
        combo_name   = '_'.join(cols) + '_'
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques  = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes    = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')

    add_gp_features(df)

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols

X,      new_cat_cols, new_num_cols = feature_engineering(X,      fit=True)
X_test, new_cat_cols, new_num_cols = feature_engineering(X_test, fit=False)
cat_cols += new_cat_cols
num_cols += new_num_cols
print("len(new_cat_cols):", len(new_cat_cols))
print("len(new_num_cols):", len(new_num_cols), "\n")

cat_cols = sorted(cat_cols)
X        = X.reindex(sorted(X.columns),           axis=1)
X_test   = X_test.reindex(sorted(X_test.columns), axis=1)
print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols), "\n")
print("X      prep shape:", X.shape)
print("X_test prep shape:", X_test.shape, "\n")

# ============================================================
# Model Components — RealMLP
# ============================================================

class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms
                      if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X: np.ndarray, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens: int, cat_dims, embed_dim: int = 8,
                 onehot_thresh: int = 8, device=None):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)])
                self.embed_layers.append(emb)
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x    = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh    = sum(onehot_dims)
            encoded     = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx+1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx+1].long()
                feat_embs.append(emb_list[model_idx](indices))
            feat_combined = torch.cat(feat_embs, dim=1)
            features.append(feat_combined)
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens: int, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias   = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 16,
                 out_dim: int = 4, freq_scale: float = 0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens      = n_ens
        self.n_features = n_features
        self.out_dim    = out_dim
        self.w1  = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1  = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2  = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2  = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic    = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat        = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim: int, cat_dims, n_numerical: int, cfg: dict):
        super().__init__()
        n_ens      = cfg["n_ens"]
        embed_dim  = cfg["embed_dim"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"])
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim   = num_emb_dim + cat_emb_dim
        act = cfg["activation"]
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(cfg["hidden_dims"]):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h
        self.hidden       = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        x     = self.hidden(torch.cat([x_num, x_cat], dim=2))
        return F.softmax(self.output_layer(x), dim=2)


# ============================================================
# Schedules · Parameter Groups · Loss · Classifier
# ============================================================

def apply_schedule(init_value: float, progress: float, sched: str,
                   flat_ratio: float = 0.3) -> float:
    if sched == "constant":
        return init_value
    elif sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    elif sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    elif sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    else:
        raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model: RealMLP, p: dict):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    LR = p["lr"]
    WD = p["weight_decay"]
    return [
        {"params": scale_p,   "lr": LR * p["lr_scale_mult"],         "weight_decay": WD * p["wd_scale_mult"],         "group": "scale"},
        {"params": pbld_p,    "lr": LR * p["pbld_lr_factor"],        "weight_decay": WD,                              "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR,                              "weight_decay": WD,                              "group": "other_w"},
        {"params": bias_p,    "lr": LR * p["lr_bias_mult"],          "weight_decay": WD * p["wd_bias_mult"],          "group": "bias"},
    ]


def smooth_ce_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                   ls: float = 0.0, class_weights: torch.Tensor = None) -> torch.Tensor:
    n_classes = y_pred.size(1)
    y_smooth  = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()


class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
            cat_col_names=None, X_test: pd.DataFrame = None):
        p             = self.params
        dev           = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose       = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num  = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat  = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr      = np.asarray(y_train)
        y_v       = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num  = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)

        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        if cat_dims:
            cat_max   = np.array(cat_dims) - 1
            X_tr_cat  = np.clip(X_tr_cat,  0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes       = np.unique(y_tr)
        self.classes_ = classes
        weights_np    = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)
        n_classes     = len(classes)

        self.model_ = RealMLP(output_dim=n_classes, cat_dims=cat_dims,
                              n_numerical=X_tr_num.shape[1], cfg=p).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        Xtn = torch.as_tensor(X_tr_num,  dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat,  dtype=torch.long,    device=dev)
        ytt = torch.as_tensor(y_tr,      dtype=torch.long,    device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long,    device=dev)

        n_ens       = p["n_ens"]
        train_bs    = p["train_bs"]
        eval_bs     = p["eval_bs"]
        epochs      = p["epochs"]
        lr_sched    = p["lr_sched"]
        flat_ratio  = p["flat_ratio"]
        ema_decay   = p["ema_decay"]
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score     = -np.inf
        best_epoch     = 0
        best_val_probs = None
        best_state     = None
        ema_state      = None
        if ema_decay > 0:
            ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress  = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start:start + train_bs]
                for g in optimizer.param_groups:
                    g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
                optimizer.zero_grad()
                y_pred   = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val   = apply_schedule(p["ls_eps"],  progress, p["ls_eps_sched"],  flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"],  flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1, n_classes),
                    ls=ls_val, class_weights=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        for key, value in self.model_.state_dict().items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)
            np.random.shuffle(train_order)

            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s:s + eval_bs], Xvc[s:s + eval_bs])
                        .mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
            epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))
            improved    = epoch_score > best_score
            if improved:
                best_score     = epoch_score
                best_epoch     = epoch + 1
                best_val_probs = val_probs.copy()
                state_src      = ema_state if ema_state is not None else self.model_.state_dict()
                best_state     = {k: v.detach().clone() for k, v in state_src.items()}
            if verbose >= 2:
                print(f"  epoch {epoch + 1}/{epochs}  score = {epoch_score:.5f}  "
                      f"best = {best_score:.5f}  ls = {ls_val:.4f}  drop = {drop_val:.4f}"
                      + (" *" if improved else ""))
            if p["use_early_stopping"]:
                patience = (best_epoch * p["early_stopping_multiplicative_patience"]
                            + p["early_stopping_additive_patience"])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch + 1} (best epoch {best_epoch})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_     = best_score
        self.best_val_probs_ = best_val_probs
        self._dev            = dev
        if verbose >= 1:
            print(f"  -> best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        eval_bs = self.params["eval_bs"]
        X_num   = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat   = np.clip(X[self.cat_col_names_].values.astype(np.int64), 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long,    device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s:s + eval_bs], Xc[s:s + eval_bs])
                    .mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ============================================================
# Config — Exact Public Baseline
# ============================================================

CONFIG = {
    "n_ens":         8,
    "embed_dim":     8,
    "onehot_thresh": 8,
    "hidden_dims":   [512, 512, 512],
    "dropout":       0.06,
    "p_drop_sched":  "expm4t",
    "activation":    nn.SiLU,
    "add_front_scale": True,
    "pbld_hidden_dim": 20,
    "pbld_out_dim":    5,
    "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor":  0.093,
    "lr":               0.01,
    "mom":              0.9,
    "sq_mom":           0.98,
    "lr_sched":        "flat_cos",
    "flat_ratio":       0.3,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult":    10.0,
    "lr_bias_mult":     0.1,
    "weight_decay":     0.013,
    "wd_scale_mult":    0.1,
    "wd_bias_mult":     0.5,
    "ema_decay":        0.997875,
    "grad_clip":        1.2,
    "ls_eps":       0.04,
    "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale"],
    "epochs":    3,
    "train_bs":  256,
    "eval_bs":   10240,
    "verbosity": 2,
    "use_early_stopping":                     False,
    "early_stopping_additive_patience":       10,
    "early_stopping_multiplicative_patience": 1,
    "device":       "cuda",
    "random_state": 63,
}

FOLDS = 7
TE    = True

# ============================================================
# 7-Fold Cross Validation
# ============================================================

def metric(y_true, y_pred):
    return balanced_accuracy_score(y_true, np.argmax(y_pred, axis=1))

skf        = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds  = np.zeros((len(X), y.nunique()))
test_preds = np.zeros((len(X_test), y.nunique()))

t_start = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr  = X.iloc[tr_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_tst = X_test.copy()

    if TE:
        te_cols  = [col for col in cat_cols if not col.endswith('bin_')]
        encoder  = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
        tr_enc   = encoder.fit_transform(X_tr[te_cols], y.iloc[tr_idx])
        val_enc  = encoder.transform(X_val[te_cols])
        tst_enc  = encoder.transform(X_tst[te_cols])
        te_names = [f"_{col}TE_class{cls}" for col in te_cols for cls in range(y.nunique())]
        X_tr[te_names]  = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc

    all_cols     = sorted(X_tr.columns.tolist())
    X_tr  = X_tr.reindex(all_cols,  axis=1)
    X_val = X_val.reindex(all_cols, axis=1)
    X_tst = X_tst.reindex(all_cols, axis=1)

    current_cat_cols = sorted([c for c in cat_cols if c in X_tr.columns])

    if fold == 1:
        print("len(FEATURES):", len(X_tr.columns.tolist()), "\n")

    bar = "━" * 46
    print(f"\n  ╔{bar}╗")
    print(f"  ║{'':>12}  FOLD  {fold:02d} / {FOLDS}  {'':>12}║")
    print(f"  ╚{bar}╝\n")

    model = RealMLP_TD_Classifier(**CONFIG)
    model.fit(
        X_tr, y.iloc[tr_idx],
        X_val, y.iloc[val_idx],
        cat_col_names=current_cat_cols,
    )
    oof_preds[val_idx] = model.best_val_probs_
    test_preds        += model.predict_proba(X_tst) / FOLDS

    fold_score = metric(y.iloc[val_idx], oof_preds[val_idx])
    print(f"\n  Fold {fold:02d} Score : {fold_score:.5f}")
    torch.cuda.empty_cache()

t_elapsed = time.time() - t_start
print("\n" + "━" * 52)
print(f"  Overall OOF Score : {metric(y, oof_preds):.5f}")
print(f"  Elapsed: {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")
print("━" * 52)

# ============================================================
# Save Outputs
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

oof_df  = pd.DataFrame({ID: train_id})
test_df = pd.DataFrame({ID: test_id})
for i, cls in enumerate(['at-risk', 'fit', 'unhealthy']):
    oof_df[cls]  = oof_preds[:, i]
    test_df[cls] = test_preds[:, i]
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'ref_realmlp_oof_preds.csv'),  index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'ref_realmlp_test_preds.csv'), index=False)

# Also save as numpy for pipeline compatibility
np.save(os.path.join(OUTPUT_DIR, 'ref_realmlp_oof.npy'), oof_preds)
np.save(os.path.join(OUTPUT_DIR, 'ref_realmlp_test.npy'), test_preds)

test_preds_cal = test_preds
sub = pd.DataFrame({ID: test_id, TARGET: np.argmax(test_preds_cal, axis=1)})
sub[TARGET] = sub[TARGET].map({0: 'at-risk', 1: 'fit', 2: 'unhealthy'})
sub.to_csv(os.path.join(OUTPUT_DIR, 'ref_realmlp_submission.csv'), index=False)

print("\n  oof_preds.csv   saved")
print("  test_preds.csv  saved")
print("  submission.csv  saved")
print()
print(sub[TARGET].value_counts().to_string())
print()
print(sub.head())
