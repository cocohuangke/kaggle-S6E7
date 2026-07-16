"""RealMLP model and feature engineering for S6E7.

Based on yekenot's 0.95090 reference notebook:
- RealMLP architecture with ensemble heads, PBLD embeddings, and categorical embeddings
- Yekenot-style feature engineering pipeline (num2cat, kbins, interactions, GP features, TE)
- RealMLP_TD_Classifier sklearn wrapper with early stopping support
- GP (genetic programming) nonlinear features: sin/sigmoid/tanh/log1p transforms

Usage:
    from pipeline.realmlp import RealMLP_TD_Classifier, realmlp_fe, CONFIG, seed_everything
"""
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Import sleep feature primitives from pipeline
from pipeline.fe import add_sleep_cat, add_sleep_interact


# ============================================================
# Seed Control
# ============================================================

def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Configuration (matches reference 0.95090 notebook)
# ============================================================
CONFIG = {
    'n_ens': 8,
    'embed_dim': 8,
    'onehot_thresh': 8,
    'hidden_dims': [512, 512, 512],
    'dropout': 0.06,
    'p_drop_sched': 'expm4t',
    'activation': nn.SiLU,
    'add_front_scale': True,
    'pbld_hidden_dim': 20,
    'pbld_out_dim': 5,
    'pbld_freq_scale': 5.0,
    'pbld_activation': nn.PReLU,
    'pbld_lr_factor': 0.093,
    'lr': 0.01,
    'mom': 0.9,
    'sq_mom': 0.98,
    'lr_sched': 'flat_cos',
    'flat_ratio': 0.3,
    'first_layer_lr_factor': 1.0,
    'first_layer_wd_factor': 0.1,
    'lr_scale_mult': 10.0,
    'lr_bias_mult': 0.1,
    'weight_decay': 0.013,
    'wd_scale_mult': 0.1,
    'wd_bias_mult': 0.5,
    'ema_decay': 0.997875,
    'grad_clip': 1.2,
    'ls_eps': 0.04,
    'ls_eps_sched': 'cos',
    'tfms': ['median_center', 'robust_scale'],
    'epochs': 3,
    'train_bs': 256,
    'eval_bs': 10240,
    'verbosity': 2,
    'use_early_stopping': False,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,
    'device': 'cuda',
    'random_state': 63,
}


# ============================================================
# Model Architecture
# ============================================================

class CategoricalFeatureLayer(nn.Module):
    """One-hot for low-cardinality; embeddings for high-cardinality."""
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
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
                self.embed_layers.append(nn.ModuleList(
                    [nn.Embedding(dim, embed_dim) for _ in range(n_ens)]))
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
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
            features.append(torch.cat(feat_embs, dim=1))
        return torch.cat(features, dim=2)


class NTPLinear(nn.Module):
    """Einsum-based linear with √(in_features) scaling per ensemble."""
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class ScalingLayer(nn.Module):
    """Per-ensemble learnable feature scaling."""
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay for numerical features."""
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4,
                 freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim-1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim-1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (
            x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)


class RealMLP(nn.Module):
    """RealMLP with ensemble heads, PBLD num embeddings, categorical embeddings."""
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        self.n_ens = n_ens

        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens, cat_dims=cat_dims, embed_dim=cfg["embed_dim"],
            onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens, n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"])

        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else cfg["embed_dim"] for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]
        act = cfg["activation"]

        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))

        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0:
                self.first_linear = linear  # explicit reference for get_parameter_groups
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]
            in_dim = out_dim_h

        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined)
        x = self.output_layer(x)
        return F.softmax(x, dim=2)


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-style preprocessing for numerical features."""
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ('median_center', 'robust_scale',
                                                'smooth_clip', 'l2_normalize')]

    def fit(self, X, y=None):
        if 'median_center' in self._tfms or 'robust_scale' in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == 'median_center':
                X -= self._median[None, :]
            elif tfm == 'robust_scale':
                X *= self._iqr_factors[None, :]
            elif tfm == 'smooth_clip':
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == 'l2_normalize':
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


# ============================================================
# Training Utilities
# ============================================================

def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None):
    """Label-smoothed cross-entropy with optional class weights."""
    n_classes = y_pred.size(1)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()


def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    """Schedule function for LR, dropout, label smoothing."""
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
    raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model, p):
    """5-group optimizer: scale, pbld, first_w, other_w, bias."""
    first_linear_weight_id = id(model.first_linear.weight)  # explicit reference
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
    LR, WD = p["lr"], p["weight_decay"]
    return [
        {"params": scale_p,   "lr": LR * p["lr_scale_mult"],         "weight_decay": WD * p["wd_scale_mult"],         "group": "scale"},
        {"params": pbld_p,    "lr": LR * p["pbld_lr_factor"],        "weight_decay": WD,                              "group": "pbld"},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": LR,                              "weight_decay": WD,                              "group": "other_w"},
        {"params": bias_p,    "lr": LR * p["lr_bias_mult"],          "weight_decay": WD * p["wd_bias_mult"],          "group": "bias"},
    ]


# ============================================================
# GP (Genetic Programming) Nonlinear Features
# ============================================================

def add_gp_features(df):
    """Add GP-discovered nonlinear transform features.

    These sin/sigmoid/tanh/log1p transforms capture nonlinear patterns
    in sleep_duration, bmi, water_intake, calorie_expenditure, step_count.
    """
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


# ============================================================
# Yekenot-style Feature Engineering
# ============================================================

def realmlp_fe(train, test, y, fe_config=None):
    """Yekenot-style feature engineering for RealMLP.

    DEPRECATED: Use pipeline.fe.get_or_create_features(fe_config, y) 
    with format='realmlp' instead. This function is kept for backward
    compatibility but will be removed in a future version.

    Matches the reference 0.95090 notebook exactly:
    - num2cat with optional extra cat features (sleep_duration_cat2_, bmi_cat1_/cat2_, heart_rate_cat2_)
    - water_intake_cat2_ with optional .round() (default: True)
    - kbins discretization
    - Interaction categories (heart_rate_bmi_, optional stress_level_physical_activity_level_)
    - GP nonlinear features (9 features, optional)

    Args:
        train, test: raw DataFrames (with id removed, target column still present in train)
        y: target series (0/1/2 integers)
        fe_config: dict with optional flags:
            - stress_pal: bool (default True) — include stress_level x physical_activity_level interaction
            - gp_features: bool (default True) — add GP nonlinear features
            - sleep_cat: bool (default False) — add sleep_duration categories
            - sleep_interact_with: list of cat cols (requires sleep_cat)
            - te: bool (default True, applied per-fold in training loop)
            - extra_cat_features: bool (default True) — create sleep_duration_cat2_, 
              bmi_cat1_/cat2_, heart_rate_cat2_ extra features. Set False for original _realmlp.
            - water_intake_round: bool (default True) — apply .round() on water_intake_cat2_.
              Set False for original _realmlp (uses .astype('int32') directly).

    Returns:
        X, X_test, y, cat_cols, num_cols
    """
    import warnings
    warnings.warn(
        "realmlp_fe() is deprecated. Use pipeline.fe.get_or_create_features() "
        "with format='realmlp' instead.",
        DeprecationWarning, stacklevel=2
    )
    TARGET = 'health_condition'
    ID = 'id'

    X = train.drop([ID, TARGET], axis=1) if TARGET in train.columns else train.copy()
    X_test = test.drop([ID], axis=1) if ID in test.columns else test.copy()

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f'  Init: {len(cat_cols)} cat + {len(num_cols)} num = {len(cat_cols)+len(num_cols)} features', flush=True)

    fe_config = fe_config or {}
    category_map = {}
    important_combos = [('heart_rate', 'bmi')]
    if fe_config.get('stress_pal', True):
        important_combos.append(('stress_level', 'physical_activity_level'))

    # Sleep discretization BEFORE NaN fill
    if fe_config.get('sleep_cat', False):
        add_sleep_cat(X, X_test)

    def _fe(df, fit=False):
        # Fill NaNs
        for col in cat_cols:
            df[col] = df[col].fillna('missing')
        for col in num_cols:
            df[col] = df[col].fillna(0.0)

        # Categorize string cats
        for col in cat_cols:
            if fit:
                codes, uniques = df[col].factorize()
                category_map[col] = uniques
            else:
                uniques = category_map[col]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = df[col].map(code_map).fillna(-1).astype('int32')
            df[col] = codes
            df[col] = df[col].astype('category')

        # Categorize numericals (num2cat) — matches reference notebook
        extra_cat = fe_config.get('extra_cat_features', True)
        water_round = fe_config.get('water_intake_round', True)
        for col in num_cols:
            cat_name = f'{col}_cat_'
            if col == 'calorie_expenditure':
                df[cat_name] = (df[col] // 5).astype('int32').astype('category')
                continue
            if col == 'water_intake':
                if water_round:
                    df['water_intake_cat2_'] = (df[col] * 50).round().astype('int32').astype('category')
                else:
                    df['water_intake_cat2_'] = (df[col] * 50).astype('int32').astype('category')
            if extra_cat:
                if col == 'sleep_duration':
                    df['sleep_duration_cat2_'] = (df[col] * 10).round().astype('int32').astype('category')
                if col == 'bmi':
                    df['bmi_cat2_'] = (df[col] - 18.5).round().astype('int32').astype('category')
                    df['bmi_cat1_'] = (24.9 - df[col]).round().astype('int32').astype('category')
                if col == 'heart_rate':
                    df['heart_rate_cat2_lo'] = (df[col] - 60).round().astype('int32').astype('category')
                    df['heart_rate_cat2_hi'] = (100 - df[col]).round().astype('int32').astype('category')
            round_level = -1
            if fit:
                round_flag = col == 'step_count'
                series = df[col].round(round_level) if round_flag else df[col]
                codes, uniques = series.factorize()
                category_map[col] = {'uniques': uniques, 'round_flag': round_flag}
            else:
                round_flag = category_map[col]['round_flag']
                uniques = category_map[col]['uniques']
                series = df[col].round(round_level) if round_flag else df[col]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = series.map(code_map).fillna(-1).astype('int32')
            df[cat_name] = codes
            df[cat_name] = df[cat_name].astype('category')

        # Discretize numericals (kbins)
        bin_config = {'sleep_duration': [70], 'water_intake': [10]}
        for col, bins_list in bin_config.items():
            for n_bins in bins_list:
                for strategy in ['quantile']:
                    bin_name = f'{col}_{n_bins}_{strategy}_bin_'
                    if fit:
                        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)
                        binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                        category_map[bin_name] = kb
                    else:
                        kb = category_map[bin_name]
                        binned = kb.transform(df[[col]]).ravel().astype('int32')
                    df[bin_name] = binned
                    df[bin_name] = df[bin_name].astype('category')

        # Interaction categories
        for cols in important_combos:
            combo_name = '_'.join(cols) + '_'
            combo_series = df[cols[0]].astype(str)
            for col in cols[1:]:
                combo_series = combo_series + '_' + df[col].astype(str)
            if fit:
                codes, uniques = pd.factorize(combo_series, sort=False)
                category_map[combo_name] = uniques
            else:
                uniques = category_map[combo_name]
                code_map = {cat: i for i, cat in enumerate(uniques)}
                codes = combo_series.map(code_map).fillna(-1).astype('int32')
            df[combo_name] = codes
            df[combo_name] = df[combo_name].astype('category')

        # GP nonlinear features
        if fe_config.get('gp_features', True):
            add_gp_features(df)

    _fe(X, fit=True)
    _fe(X_test, fit=False)

    # Post-_fe: factorize sleep_cat_ and add sleep interactions
    if fe_config.get('sleep_cat', False):
        codes, uniques = X['sleep_cat_'].factorize()
        category_map['sleep_cat_'] = uniques
        X['sleep_cat_'] = codes; X['sleep_cat_'] = X['sleep_cat_'].astype('category')
        code_map = {cat: i for i, cat in enumerate(uniques)}
        X_test['sleep_cat_'] = X_test['sleep_cat_'].map(code_map).fillna(-1).astype('int32')
        X_test['sleep_cat_'] = X_test['sleep_cat_'].astype('category')

        interact_cols = fe_config.get('sleep_interact_with', [])
        if interact_cols:
            stress_pal_name = 'stress_level_physical_activity_level_'
            resolved = [stress_pal_name if c in ('stress_pal', 'stress_level_physical_activity_level_') else c
                       for c in interact_cols]
            sleep_interact_cols = add_sleep_interact(X, X_test, resolved)
            for col in sleep_interact_cols:
                codes, uniques = X[col].factorize()
                category_map[col] = uniques
                X[col] = codes; X[col] = X[col].astype('category')
                code_map = {cat: i for i, cat in enumerate(uniques)}
                X_test[col] = X_test[col].map(code_map).fillna(-1).astype('int32')
                X_test[col] = X_test[col].astype('category')

    # Collect new features
    new_cat_cols = [col for col in X.columns if col.endswith('_')]
    new_num_cols = [col for col in X.columns if col.startswith('_gp_')]
    cat_cols += new_cat_cols
    num_cols += new_num_cols
    cat_cols = sorted(cat_cols)
    X = X.reindex(sorted(X.columns), axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)

    print(f'  After FE: {len(cat_cols)} cat + {len(num_cols)} num = {X.shape[1]} features', flush=True)
    return X, X_test, y, cat_cols, num_cols


# ============================================================
# RealMLP_TD_Classifier (sklearn wrapper)
# ============================================================

class RealMLP_TD_Classifier(BaseEstimator):
    """Sklearn-compatible RealMLP classifier with EMA and early stopping.

    Matches the reference 0.95090 notebook's classifier exactly.
    """
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
        p = self.params
        dev = torch.device(p['device'] if torch.cuda.is_available() else 'cpu')
        verbose = p['verbosity']
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train)
        y_v = np.asarray(y_val)

        self.preprocessor_ = NumericalPreprocessor(p['tfms'])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
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
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
            X_val_cat = np.clip(X_val_cat, 0, cat_max)

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)
        n_classes = len(classes)

        self.model_ = RealMLP(output_dim=n_classes, cat_dims=cat_dims,
                              n_numerical=X_tr_num.shape[1], cfg=p).to(dev)

        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups:
            g['lr_base'] = g['lr']
        optimizer = torch.optim.AdamW(param_groups, betas=(p['mom'], p['sq_mom']))

        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

        n_ens = p['n_ens']
        train_bs = p['train_bs']
        eval_bs = p['eval_bs']
        epochs = p['epochs']
        lr_sched = p['lr_sched']
        flat_ratio = p['flat_ratio']
        ema_decay = p['ema_decay']
        total_steps = epochs * len(y_tr)
        train_order = np.arange(len(y_tr))

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        best_state = None
        ema_state = None
        if ema_decay > 0:
            ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start:start + train_bs]
                for g in optimizer.param_groups:
                    g['lr'] = apply_schedule(g['lr_base'], progress, lr_sched, flat_ratio)
                optimizer.zero_grad()
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val = apply_schedule(p['ls_eps'], progress, p['ls_eps_sched'], flat_ratio)
                drop_val = apply_schedule(p['dropout'], progress, p['p_drop_sched'], flat_ratio)
                for dm in self.model_._dropout_modules:
                    dm.p = drop_val
                loss = smooth_ce_loss(
                    ytt[idx_batch].repeat_interleave(n_ens),
                    y_pred.reshape(-1, n_classes),
                    ls=ls_val, class_weights=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p['grad_clip'])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        for key, value in self.model_.state_dict().items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)
            np.random.shuffle(train_order)

            # Validation (with EMA)
            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate([
                    self.model_(Xvn[s:s+eval_bs], Xvc[s:s+eval_bs]).mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_v), eval_bs)
                ], axis=0)
            epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = val_probs.copy()
                state_src = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in state_src.items()}
            if verbose >= 2:
                print(f"  epoch {epoch+1}/{epochs}  score = {epoch_score:.5f}  "
                      f"best = {best_score:.5f}  ls = {ls_val:.4f}  drop = {drop_val:.4f}"
                      + (" *" if improved else ""), flush=True)
            if p['use_early_stopping']:
                patience = (best_epoch * p['early_stopping_multiplicative_patience']
                            + p['early_stopping_additive_patience'])
                if (epoch + 1) > patience:
                    if verbose >= 1:
                        print(f"  Early stopping at epoch {epoch+1} (best epoch {best_epoch})", flush=True)
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_ = best_score
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"  -> best score: {best_score:.5f}  (epoch {best_epoch})", flush=True)
        return self

    def predict_proba(self, X):
        eval_bs = self.params['eval_bs']
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = np.clip(X[self.cat_col_names_].values.astype(np.int64), 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                self.model_(Xn[s:s+eval_bs], Xc[s:s+eval_bs]).mean(dim=1).cpu().numpy()
                for s in range(0, len(X_num), eval_bs)
            ], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ============================================================
# Single-Fold Training (legacy, kept for backward compat)
# ============================================================

def train_realmlp_fold(X_tr, y_tr, X_val, y_val, cat_col_names, num_col_names,
                       X_tst=None, p=None, device='cuda'):
    """Train RealMLP for a single fold. Legacy interface — prefer RealMLP_TD_Classifier."""
    p = p or CONFIG
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    n_classes = len(np.unique(y_tr))

    X_tr_num = X_tr[num_col_names].values.astype(np.float32)
    X_val_num = X_val[num_col_names].values.astype(np.float32)
    X_tr_cat = X_tr[cat_col_names].values.astype(np.int64)
    X_val_cat = X_val[cat_col_names].values.astype(np.int64)

    preprocessor = NumericalPreprocessor(p['tfms'])
    preprocessor.fit(X_tr_num)
    X_tr_num = preprocessor.transform(X_tr_num)
    X_val_num = preprocessor.transform(X_val_num)

    # Cat dims
    all_cat = [X_tr_cat, X_val_cat]
    if X_tst is not None:
        X_tst_cat = X_tst[cat_col_names].values.astype(np.int64)
        all_cat.append(X_tst_cat)
    cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
    cat_max = np.array(cat_dims) - 1
    X_tr_cat = np.clip(X_tr_cat, 0, cat_max)
    X_val_cat = np.clip(X_val_cat, 0, cat_max)
    if X_tst is not None:
        X_tst_cat = np.clip(X_tst_cat, 0, cat_max)

    # Class weights
    classes = np.unique(y_tr)
    weights_np = compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
    class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

    # Build model
    model = RealMLP(output_dim=n_classes, cat_dims=cat_dims, n_numerical=X_tr_num.shape[1], cfg=p).to(dev)
    param_groups = get_parameter_groups(model, p)
    for g in param_groups:
        g['lr_base'] = g['lr']
    optimizer = torch.optim.AdamW(param_groups, betas=(p['mom'], p['sq_mom']))

    Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
    Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
    ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
    Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
    Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)

    if X_tst is not None:
        X_tst_num = preprocessor.transform(X_tst[num_col_names].values.astype(np.float32))
        Xtn_test = torch.as_tensor(X_tst_num, dtype=torch.float32, device=dev)
        Xtc_test = torch.as_tensor(X_tst_cat, dtype=torch.long, device=dev)

    n_ens = p['n_ens']
    train_bs = p['train_bs']
    eval_bs = p['eval_bs']
    epochs = p['epochs']
    total_steps = epochs * len(y_tr)
    train_order = np.arange(len(y_tr))

    ema_state = None
    if p['ema_decay'] > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    best_score, best_epoch, best_val_probs, best_state = -1.0, 0, None, None

    for epoch in range(epochs):
        model.train()
        for start in range(0, len(y_tr), train_bs):
            progress = (epoch * len(y_tr) + start) / total_steps
            idx_batch = train_order[start:start + train_bs]
            for g in optimizer.param_groups:
                g['lr'] = apply_schedule(g['lr_base'], progress, p['lr_sched'], p['flat_ratio'])
            optimizer.zero_grad()
            y_pred = model(Xtn[idx_batch], Xtc[idx_batch])
            ls_val = apply_schedule(p['ls_eps'], progress, p['ls_eps_sched'], p['flat_ratio'])
            drop_val = apply_schedule(p['dropout'], progress, p['p_drop_sched'], p['flat_ratio'])
            for dm in model._dropout_modules:
                dm.p = drop_val
            loss = smooth_ce_loss(
                ytt[idx_batch].repeat_interleave(n_ens), y_pred.reshape(-1, n_classes),
                ls=ls_val, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p['grad_clip'])
            optimizer.step()
            if ema_state is not None:
                with torch.no_grad():
                    ms = model.state_dict()
                    for key, value in ms.items():
                        if torch.is_floating_point(value):
                            ema_state[key].mul_(p['ema_decay']).add_(value.detach(), alpha=1.0 - p['ema_decay'])
                        else:
                            ema_state[key].copy_(value)
        np.random.shuffle(train_order)

        # Validation (with EMA)
        model.eval()
        if ema_state is not None:
            live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state, strict=True)
        with torch.no_grad():
            val_probs = np.concatenate([
                model(Xvn[s:s+eval_bs], Xvc[s:s+eval_bs]).mean(dim=1).cpu().numpy()
                for s in range(0, len(y_val), eval_bs)
            ], axis=0)
        epoch_score = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
        if epoch_score > best_score:
            best_score, best_epoch = epoch_score, epoch + 1
            best_val_probs = val_probs.copy()
            state_src = ema_state if ema_state is not None else model.state_dict()
            best_state = {k: v.detach().clone() for k, v in state_src.items()}
        if ema_state is not None:
            model.load_state_dict(live_state, strict=True)
        if p['verbosity'] >= 2:
            print(f'    epoch {epoch+1}/{epochs}  score={epoch_score:.5f}  best={best_score:.5f}', flush=True)

    model.load_state_dict(best_state, strict=True)

    # Test predictions
    tst_probs = None
    if X_tst is not None:
        model.eval()
        with torch.no_grad():
            tst_probs = np.concatenate([
                model(Xtn_test[s:s+eval_bs], Xtc_test[s:s+eval_bs]).mean(dim=1).cpu().numpy()
                for s in range(0, len(X_tst), eval_bs)
            ], axis=0)

    return best_val_probs, tst_probs, best_score, best_epoch
