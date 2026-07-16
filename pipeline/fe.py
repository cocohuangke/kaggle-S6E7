"""Feature engineering pipeline for S6E7.

Config-driven: each FE config has a unique tag → cache/fe/{tag}.parquet
If cache exists with same tag, skip FE and load directly.

Design:
- FE parameters encoded in tag name (human-readable)
- FE outputs: (X_train, X_test, cat_indices, feature_names) saved as parquet + json metadata
- Pipeline reads FE by tag; if cache miss, runs FE and saves
"""
import json
import os
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.model_selection import StratifiedKFold

# Column definitions
NUM_COLS = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
            'step_count', 'exercise_duration', 'water_intake']
CAT_COLS = ['diet_type', 'stress_level', 'sleep_quality',
            'physical_activity_level', 'smoking_alcohol', 'gender']

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'fe')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# ============================================================
# FE primitive functions (each does one thing, composable)
# ============================================================

def add_num2cat(train, test):
    """A: Convert numerics to categories (yekenot style).
    - calorie_expenditure // 5
    - water_intake * 50 (then int)
    - step_count round(-1)
    Must be called BEFORE fill missing, so NaN -> 'missing' category.
    """
    for df in (train, test):
        # Handle NaN separately to avoid IntCastingNaNError
        cal = df['calorie_expenditure'].copy()
        wat = df['water_intake'].copy()
        step = df['step_count'].copy()

        cal_mask = cal.isna()
        wat_mask = wat.isna()
        step_mask = step.isna()

        # Fill NaN temporarily for type conversion (match original: int64)
        cal_filled = cal.fillna(0)
        wat_filled = wat.fillna(0)
        step_filled = step.fillna(0)

        df['calorie_cat_'] = np.where(cal_mask, 'missing',
                                       (cal_filled.astype('int64') // 5).astype(str))
        df['water_cat2_'] = np.where(wat_mask, 'missing',
                                      (wat_filled.astype('int64') * 50).astype(str))
        df['step_cat_'] = np.where(step_mask, 'missing',
                                    (step_filled.round(-1).astype('int64')).astype(str))
    return ['calorie_cat_', 'water_cat2_', 'step_cat_']


def add_gp_features(train, test):
    """Add GP-discovered nonlinear transform features (9 features).

    sin/sigmoid/tanh/log1p transforms capture nonlinear patterns
    in sleep_duration, bmi, water_intake, calorie_expenditure, step_count.

    For format='realmlp': called after NaN fill (0.0), returns float32 columns.
    """
    new_cols = []
    for df in (train, test):
        sd  = df['sleep_duration'].values.astype(np.float64)
        bmi = df['bmi'].values.astype(np.float64)
        wi  = df['water_intake'].values.astype(np.float64)
        cal = df['calorie_expenditure'].values.astype(np.float64)
        sc  = df['step_count'].values.astype(np.float64)
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
        if not new_cols:
            new_cols = list(gp.keys())
    return new_cols


def add_extra_cat_features(train, test, water_intake_round=True):
    """Add extra categorical features for RealMLP format.

    Creates: sleep_duration_cat2_, bmi_cat1_, bmi_cat2_,
             heart_rate_cat2_lo, heart_rate_cat2_hi

    Must be called AFTER NaN fill (RealMLP: 0.0 for num).
    These are int32 → category dtype columns.
    """
    new_cols = []
    for df in (train, test):
        if 'sleep_duration_cat2_' not in df.columns:
            df['sleep_duration_cat2_'] = (df['sleep_duration'] * 10).round().astype('int32')
        if 'bmi_cat2_' not in df.columns:
            df['bmi_cat2_'] = (df['bmi'] - 18.5).round().astype('int32')
        if 'bmi_cat1_' not in df.columns:
            df['bmi_cat1_'] = (24.9 - df['bmi']).round().astype('int32')
        if 'heart_rate_cat2_lo' not in df.columns:
            df['heart_rate_cat2_lo'] = (df['heart_rate'] - 60).round().astype('int32')
        if 'heart_rate_cat2_hi' not in df.columns:
            df['heart_rate_cat2_hi'] = (100 - df['heart_rate']).round().astype('int32')
    new_cols = ['sleep_duration_cat2_', 'bmi_cat2_', 'bmi_cat1_',
                'heart_rate_cat2_lo', 'heart_rate_cat2_hi']
    return new_cols


def add_stress_pal(train, test):
    """Create stress_level + physical_activity_level interaction.
    Must be called BEFORE fill missing for NaN -> 'MISSING'.
    """
    for df in (train, test):
        sl = df['stress_level'].fillna('MISSING').astype(str)
        pal = df['physical_activity_level'].fillna('MISSING').astype(str)
        df['stress_pal'] = sl + '_' + pal
    return ['stress_pal']


def add_sleep_cat(train, test):
    """Sleep duration threshold categories: <6, 6-7, >=7, missing.
    Based on EDA: sleep<6 → unhealthy boundary, sleep>=7 → fit boundary.
    Must be called BEFORE fill missing so NaN -> 'missing' category.
    """
    for df in (train, test):
        sd = df['sleep_duration']
        mask = sd.isna()
        cat = pd.Series('missing', index=df.index, dtype=object)
        cat[~mask & (sd < 6)] = 'lt6'
        cat[~mask & (sd >= 6) & (sd < 7)] = '6to7'
        cat[~mask & (sd >= 7)] = 'ge7'
        df['sleep_cat_'] = cat
    return ['sleep_cat_']


def add_sleep_interact(train, test, cols):
    """Cartesian product of sleep_cat_ with each column in cols.
    
    Requires sleep_cat: true (add_sleep_cat called first).
    Must be called BEFORE fill missing for NaN handling.
    
    Args:
        cols: list of column names to interact with sleep_cat_
    Returns:
        list of new column names (sleep_cat_X_{col}_)
    """
    new_cols = []
    for col in cols:
        new_name = f'sleep_cat_X_{col}_'
        for df in (train, test):
            sc = df['sleep_cat_'].fillna('missing').astype(str)
            cv = df[col].fillna('missing').astype(str)
            df[new_name] = sc + '_X_' + cv
        new_cols.append(new_name)
    return new_cols


def add_stress_bin(train, test):
    """Stress level binary indicators: is_high_stress, is_low_stress.
    Must be called BEFORE fill missing so NaN -> 0 (missing stress is not high/low).
    """
    for df in (train, test):
        sl = df['stress_level']
        df['is_high_stress'] = (sl == 'high').astype(int)
        df['is_low_stress'] = (sl == 'low').astype(int)
    return ['is_high_stress', 'is_low_stress']


def add_key_rule3(train, test):
    """Three-feature interaction: stress_level + physical_activity_level + sleep_cat_.
    Encodes the generation rule: (low,active,ge7)->fit, (high,*,lt6)->unhealthy, etc.
    Requires sleep_cat to be already created (call add_sleep_cat first).
    Must be called BEFORE fill missing for NaN handling.
    """
    if 'sleep_cat_' not in train.columns:
        raise ValueError("add_key_rule3 requires 'sleep_cat_' column. "
                         "Enable sleep_cat in fe_config before key_rule3.")
    for df in (train, test):
        sl = df['stress_level'].fillna('MISSING').astype(str)
        pal = df['physical_activity_level'].fillna('MISSING').astype(str)
        sc = df['sleep_cat_'].fillna('missing').astype(str)
        df['key_rule3_'] = sl + '_' + pal + '_' + sc
    return ['key_rule3_']


def fill_missing(train, test, num_strategy='median'):
    """Fill missing values."""
    if num_strategy == 'median':
        num_fills = {col: train[col].median() for col in NUM_COLS}
    else:
        num_fills = {col: 0.0 for col in NUM_COLS}
    for col in NUM_COLS:
        train[col] = train[col].fillna(num_fills[col])
        test[col] = test[col].fillna(num_fills[col])
    for col in CAT_COLS:
        train[col] = train[col].fillna('missing')
        test[col] = test[col].fillna('missing')


def add_kbins(train, test):
    """B: KBinsDiscretizer (sleep 70 bins, water 10 bins)."""
    from sklearn.preprocessing import KBinsDiscretizer
    new_cols = []
    for col, n_bins in [('sleep_duration', 70), ('water_intake', 10)]:
        name = f'{col}_{n_bins}qbin_'
        kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                              strategy='quantile', subsample=None)
        train[name] = kb.fit_transform(train[[col]]).ravel().astype('int32').astype(str)
        test[name] = kb.transform(test[[col]]).ravel().astype('int32').astype(str)
        new_cols.append(name)
    return new_cols


def add_heart_bmi(train, test):
    """C: heart_rate x bmi interaction (binned).
    Uses train quantiles for both train and test to avoid distribution mismatch.
    """
    for col, bin_name in [('heart_rate', 'hr_bin_'), ('bmi', 'bmi_bin_')]:
        # Compute bin edges from train only
        _, bin_edges = pd.qcut(train[col], q=10, retbins=True, duplicates='drop')
        # Add margins to include test values outside train range
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        train[bin_name] = pd.cut(train[col], bins=bin_edges, labels=False,
                                 include_lowest=True).astype(str)
        test[bin_name] = pd.cut(test[col], bins=bin_edges, labels=False,
                                include_lowest=True).astype(str)
    train['heart_bmi_'] = train['hr_bin_'] + '_' + train['bmi_bin_']
    test['heart_bmi_'] = test['hr_bin_'] + '_' + test['bmi_bin_']
    return ['hr_bin_', 'bmi_bin_', 'heart_bmi_']


def target_encode(train, test, y, te_cols, n_splits=5, random_state=None):
    """D: Per-fold target encoding (avoids leakage)."""
    if random_state is None:
        random_state = n_splits * 42  # deterministic but configurable
    n_classes = len(np.unique(y))
    te_names = [f"_{col}TE_c{c}" for col in te_cols for c in range(n_classes)]

    for name in te_names:
        train[name] = 0.0
        test[name] = 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, y)):
        enc = TargetEncoder(cv=n_splits, smooth='auto', target_type='multiclass',
                            shuffle=True, random_state=random_state)
        enc.fit_transform(train.iloc[trn_idx][te_cols], y[trn_idx])
        val_enc = enc.transform(train.iloc[val_idx][te_cols])
        if hasattr(val_enc, 'values'):
            val_enc = val_enc.values
        for i, name in enumerate(te_names):
            train.iloc[val_idx, train.columns.get_loc(name)] = val_enc[:, i]

    enc_full = TargetEncoder(cv=n_splits, smooth='auto', target_type='multiclass',
                             shuffle=True, random_state=random_state)
    enc_full.fit(train[te_cols], y)
    tst_enc = enc_full.transform(test[te_cols])
    if hasattr(tst_enc, 'values'):
        tst_enc = tst_enc.values
    for i, name in enumerate(te_names):
        test[name] = tst_enc[:, i]

    return te_names


def target_encode_numeric(train, test, y, n_splits=5, random_state=None):
    """Num TE: Target encoding on numeric columns as strings.

    7 numeric columns -> 21 TE features (7 x 3 classes).
    Uses the same TargetEncoder approach as cat TE but operates on
    numeric columns converted to strings.
    """
    if random_state is None:
        random_state = n_splits * 42
    n_classes = len(np.unique(y))
    te_num_names = [f'_{col}numTE_class{cls}' for col in NUM_COLS for cls in range(n_classes)]

    for name in te_num_names:
        train[name] = 0.0
        test[name] = 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_num_str = train[NUM_COLS].astype(str).fillna('na')
    test_num_str = test[NUM_COLS].astype(str).fillna('na')

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, y)):
        enc = TargetEncoder(cv=n_splits, smooth='auto', target_type='multiclass',
                            shuffle=True, random_state=random_state)
        enc.fit(train_num_str.iloc[trn_idx], y[trn_idx])
        val_enc = enc.transform(train_num_str.iloc[val_idx])
        if hasattr(val_enc, 'values'):
            val_enc = val_enc.values
        for i, name in enumerate(te_num_names):
            train.loc[val_idx, name] = val_enc[:, i]

    enc_full = TargetEncoder(cv=n_splits, smooth='auto', target_type='multiclass',
                             shuffle=True, random_state=random_state)
    enc_full.fit(train_num_str, y)
    tst_enc = enc_full.transform(test_num_str)
    if hasattr(tst_enc, 'values'):
        tst_enc = tst_enc.values
    for i, name in enumerate(te_num_names):
        test[name] = tst_enc[:, i]

    return te_num_names


def label_encode(train, test, cat_cols):
    """Label encode categorical columns (alphabetical order)."""
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = sorted(set(
            train[col].astype(str).unique().tolist() +
            test[col].astype(str).unique().tolist()
        ))
        le.fit(all_vals)
        le_dict[col] = le
    for col in cat_cols:
        train[col] = le_dict[col].transform(train[col].astype(str))
        test[col] = le_dict[col].transform(test[col].astype(str))
    return le_dict


# Ordinal encoding maps: higher value = healthier/better
ORDINAL_MAPS = {
    'diet_type': {'veg': 0, 'non-veg': 1, 'balanced': 2},
    'stress_level': {'high': 0, 'medium': 1, 'low': 2},
    'sleep_quality': {'poor': 0, 'average': 1, 'good': 2},
    'physical_activity_level': {'sedentary': 0, 'moderate': 1, 'active': 2},
    'smoking_alcohol': {'yes': 0, 'occasional': 1, 'no': 2},
    'gender': {'male': 0, 'female': 1, 'other': 2},
}


def ordinal_encode(train, test, cat_cols):
    """Ordinal encode categorical columns with health-ordered mappings.

    Higher value = healthier/better state. This preserves monotonic
    health information that LabelEncoder (alphabetical) loses.

    For example: diet_type balanced=2 > non-veg=1 > veg=0,
    whereas LabelEncoder would give balanced=0, non-veg=1, veg=2.

    Must be called BEFORE fill_missing. NaN values are preserved as NaN
    (models like LGBM/XGB handle NaN natively). After fill_missing,
    any remaining NaN in cat cols will be filled with -1 by fill_missing
    (if ordinal_encode is enabled, fill_missing skips ordinal-encoded cols).
    """
    for col in cat_cols:
        if col in ORDINAL_MAPS:
            m = ORDINAL_MAPS[col]
            train[col] = train[col].map(m)
            test[col] = test[col].map(m)
            # NaN stays NaN — models handle natively
            # Cast to float (matches reference notebook)
            train[col] = train[col].astype('float')
            test[col] = test[col].astype('float')
        else:
            # Fallback to LabelEncoder for columns without ordinal maps
            le = LabelEncoder()
            all_vals = sorted(set(
                train[col].astype(str).unique().tolist() +
                test[col].astype(str).unique().tolist()
            ))
            le.fit(all_vals)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))


def add_domain_features(train, test, variant='full', replace_inf=True):
    """Add domain-knowledge health features (from 0.95043 reference notebook).

    Creates 13 (full) or 9 (xgb) new numeric features based on medical/health
    domain knowledge. These are the top-2 most important features in the
    reference LGBM model.

    Args:
        variant: 'full' (13 features for LGBM) or 'xgb' (9 features, omits
                 speed_score, hydration_score, depression_score, no_exercises)
        replace_inf: If True (default), replace inf→NaN for XGB compatibility.
            Set False to match reference notebook which keeps inf (LGBM handles
            inf natively). Only affects ratio features (calorie_score, steps_score,
            speed_score, hydration_score).

    Returns:
        list of new column names
    """
    new_cols = []
    for df in (train, test):
        # healthy_score: mean of 5 ordinal-encoded health columns
        health_cols = ['diet_type', 'stress_level', 'sleep_quality',
                       'physical_activity_level', 'smoking_alcohol']
        # Use ordinal values if already encoded (float), else map
        for col in health_cols:
            if col not in ORDINAL_MAPS and df[col].dtype == object:
                # Not yet ordinal-encoded, apply mapping inline
                pass  # will use current values
        df['healthy_score'] = df[health_cols].mean(axis=1)
        new_cols.append('healthy_score')

        # sleep_score: gaussian(sleep_duration) × sleep_quality multiplier
        def gauss(x):
            return np.exp(-(x - 8) ** 2 / 4.5)
        sleep_score_map = {0: 0.5, 1: 1.0, 2: 1.5}
        df['sleep_score'] = gauss(df['sleep_duration']) * df['sleep_quality'].map(sleep_score_map)
        new_cols.append('sleep_score')

        # calorie_score: calorie_expenditure / bmi
        df['calorie_score'] = df['calorie_expenditure'] / df['bmi']
        if replace_inf:
            df['calorie_score'] = df['calorie_score'].replace([np.inf, -np.inf], np.nan)
        new_cols.append('calorie_score')

        # steps_score: step_count / bmi
        df['steps_score'] = df['step_count'] / df['bmi']
        if replace_inf:
            df['steps_score'] = df['steps_score'].replace([np.inf, -np.inf], np.nan)
        new_cols.append('steps_score')

        # sport_score: log1p(step_count × exercise_duration × activity_map)
        activity_map = {0: 0.5, 1: 1.0, 2: 1.5}
        df['sport_score'] = np.log1p(
            df['step_count'] * df['exercise_duration'] * df['physical_activity_level'].map(activity_map)
        )
        new_cols.append('sport_score')

        if variant == 'full':
            # speed_score: step_count / exercise_duration
            df['speed_score'] = df['step_count'] / df['exercise_duration']
            if replace_inf:
                df['speed_score'] = df['speed_score'].replace([np.inf, -np.inf], np.nan)
            new_cols.append('speed_score')

            # hydration_score: water_intake / calorie_expenditure
            df['hydration_score'] = df['water_intake'] / df['calorie_expenditure']
            if replace_inf:
                df['hydration_score'] = df['hydration_score'].replace([np.inf, -np.inf], np.nan)
            new_cols.append('hydration_score')

            # depression_score: mean(smoking_alcohol, stress_level)
            df['depression_score'] = df[['smoking_alcohol', 'stress_level']].mean(axis=1)
            new_cols.append('depression_score')

            # no_exercises: exercise_duration == 0 flag (NaN preserved)
            df['no_exercises'] = np.where(df['exercise_duration'].isna(), np.nan,
                                          (df['exercise_duration'] == 0).astype(int))
            new_cols.append('no_exercises')

        # bradycardia: heart_rate < 60 (NaN preserved)
        df['bradycardia'] = np.where(df['heart_rate'].isna(), np.nan,
                                     (df['heart_rate'] < 60).astype(int))
        new_cols.append('bradycardia')

        # tachycardia: heart_rate > 100 (NaN preserved)
        df['tachycardia'] = np.where(df['heart_rate'].isna(), np.nan,
                                     (df['heart_rate'] > 100).astype(int))
        new_cols.append('tachycardia')

        # underweight: bmi < 18.5 (NaN preserved)
        df['underweight'] = np.where(df['bmi'].isna(), np.nan,
                                     (df['bmi'] < 18.5).astype(int))
        new_cols.append('underweight')

        # overweight: bmi > 25 (NaN preserved)
        df['overweight'] = np.where(df['bmi'].isna(), np.nan,
                                    (df['bmi'] > 25).astype(int))
        new_cols.append('overweight')

    # Deduplicate (both train and test produce same col names)
    new_cols = list(dict.fromkeys(new_cols))
    return new_cols


# ============================================================
# Config-driven FE builder
# ============================================================

def build_tag(fe_config):
    """Generate human-readable tag from FE config dict.

    Tag encodes all FE parameters so same config → same tag → cache hit.
    Format: {features}_{num_fill}_{nfolds}f{_te?}_{ndim}d
    e.g.: A_median_5f_17d, ABC_zero_5f_te_69d
    """
    parts = []
    # Feature flags
    flags = ''
    if fe_config.get('num2cat', False):
        flags += 'A'
    if fe_config.get('kbins', False):
        flags += 'B'
    if fe_config.get('heart_bmi', False):
        flags += 'C'
    if fe_config.get('te', False):
        flags += 'D'
    if fe_config.get('sleep_cat', False):
        flags += 'S'
    if fe_config.get('sleep_interact_with', None):
        flags += 'I'
    if fe_config.get('stress_pal', True):
        flags += 'P'
    if fe_config.get('stress_bin', False):
        flags += 'H'
    if fe_config.get('key_rule3', False):
        flags += 'K'
    if fe_config.get('num_te', False):
        flags += 'N'
    if fe_config.get('gp_features', False):
        flags += 'G'
    if fe_config.get('extra_cat_features', False):
        flags += 'E'
    if fe_config.get('format', 'gbdt') == 'realmlp':
        flags += 'R'
    if fe_config.get('ordinal_encode', False):
        flags += 'O'
    if fe_config.get('domain_features', False):
        variant = fe_config.get('domain_features_variant', 'full')
        flags += 'F' if variant == 'full' else 'Fx'  # F=full(13), Fx=xgb(9)
    if fe_config.get('col_order', 'default') == 'cat_first':
        flags += 'W'  # W=cat-first column order (reference notebook convention)
    if not fe_config.get('replace_inf', True):
        flags += 'V'  # V=keep inf (not replaced to NaN)
    if not flags:
        flags = 'base'
    parts.append(flags)

    # Num fill strategy
    parts.append(fe_config.get('num_fill', 'median'))

    # TE fold count (only if TE enabled)
    if fe_config.get('te', False) or fe_config.get('num_te', False):
        parts.append(f'te{fe_config.get("te_folds", 5)}f')

    # Dropped features (hash to avoid truncation collision)
    drop = fe_config.get('drop_features', [])
    if drop:
        drop_hash = hashlib.md5(','.join(sorted(drop)).encode()).hexdigest()[:6]
        parts.append(f'd{drop_hash}')

    return '_'.join(parts)


def compute_ndim(fe_config):
    """Compute total feature dimensionality from config."""
    n = len(NUM_COLS)  # 7 numeric
    # Categorical: base 6 + optional stress_pal
    n_cat = len(CAT_COLS)  # 6 base cat
    if fe_config.get('stress_pal', True):
        n_cat += 1  # stress_pal
    if fe_config.get('num2cat', False):
        n_cat += 3  # calorie_cat_, water_cat2_, step_cat_
    if fe_config.get('kbins', False):
        n_cat += 2  # sleep 70qbin, water 10qbin
    if fe_config.get('heart_bmi', False):
        n_cat += 3  # hr_bin, bmi_bin, heart_bmi
    if fe_config.get('sleep_cat', False):
        n_cat += 1  # sleep_cat_
    if fe_config.get('sleep_interact_with', None):
        n_cat += len(fe_config['sleep_interact_with'])  # sleep_cat_X_{col}_ per col
    if fe_config.get('key_rule3', False):
        n_cat += 1  # key_rule3_
    n_extra_num = 0
    if fe_config.get('stress_bin', False):
        n_extra_num += 2  # is_high_stress, is_low_stress
    if fe_config.get('gp_features', False):
        n_extra_num += 9  # 9 GP nonlinear features
    if fe_config.get('extra_cat_features', False):
        n_cat += 5  # sleep_duration_cat2_, bmi_cat1_, bmi_cat2_, heart_rate_cat2_lo, heart_rate_cat2_hi
    if fe_config.get('domain_features', False):
        variant = fe_config.get('domain_features_variant', 'full')
        n_extra_num += 13 if variant == 'full' else 9  # domain health features

    n_te = 0
    if fe_config.get('te', False):
        # TE on all cat cols except bin cols
        te_cat_count = n_cat  # all cat cols get TE in current design
        n_te = te_cat_count * 3  # 3 classes
    if fe_config.get('num_te', False):
        n_te += len(NUM_COLS) * 3  # 7 numeric cols x 3 classes = 21

    total = n + n_extra_num + n_cat + n_te

    # Subtract dropped features
    drop = fe_config.get('drop_features', [])
    if drop:
        # Count how many of the dropped features are in the final set
        all_feats = NUM_COLS + (['is_high_stress', 'is_low_stress'] if fe_config.get('stress_bin', False) else [])
        # Can't precisely count without running FE, so just subtract len(drop) as approximation
        total -= len(drop)

    return total


def get_feature_subset(X_train, X_test, cat_indices, feature_names, subset):
    """Filter feature matrix to a specified column subset.

    Args:
        X_train, X_test: full feature DataFrames
        cat_indices: list of categorical column indices (0-based)
        feature_names: list of all feature column names
        subset: 'all' (default), 'base' (NUM+CAT only, no TE columns)

    Returns:
        (X_train_sub, X_test_sub, cat_indices_sub)
    """
    if subset is None or subset == 'all':
        return X_train, X_test, cat_indices

    if subset == 'base':
        # Keep only columns that are NOT TE features
        keep_cols = [c for c in feature_names if 'TE' not in c]
        X_train_sub = X_train[keep_cols]
        X_test_sub = X_test[keep_cols]
        # cat_indices stay valid since TE cols are always appended after cat cols
        return X_train_sub, X_test_sub, cat_indices

    if isinstance(subset, list):
        X_train_sub = X_train[subset]
        X_test_sub = X_test[subset]
        cat_col_names = set(feature_names[i] for i in cat_indices)
        cat_indices_sub = [i for i, c in enumerate(subset) if c in cat_col_names]
        return X_train_sub, X_test_sub, cat_indices_sub

    raise ValueError(f'Unknown feature_subset: {subset}')


def _realmlp_fe_transform(train, test, fe_config):
    """RealMLP-style feature engineering pipeline.

    Output: DataFrame with category dtype columns + numeric columns.
    Returns (X_train, X_test, cat_col_names, feature_names, category_map).

    Key differences from GBDT format:
    - factorize → category dtype (not LabelEncoder → ints)
    - NO pre-computed TE (RealMLP does per-fold TE in training loop)
    - num2cat uses factorize + direct int32→category (not string→LabelEncoder)
    - Includes GP features, extra cat features, water_intake_cat2_ with .round()
    """
    from sklearn.preprocessing import KBinsDiscretizer
    extra_cat_flag = fe_config.get('extra_cat_features', True)
    gp_flag = fe_config.get('gp_features', True)
    water_round = fe_config.get('water_intake_round', True)
    stress_pal_flag = fe_config.get('stress_pal', True)

    category_map = {}
    all_cat_cols = list(CAT_COLS)  # track all cat cols for return

    # Step 0: Fill NaN (RealMLP style: 0.0 for num, 'missing' for cat)
    for col in NUM_COLS:
        train[col] = train[col].fillna(0.0)
        test[col] = test[col].fillna(0.0)
    for col in CAT_COLS:
        train[col] = train[col].fillna('missing')
        test[col] = test[col].fillna('missing')

    # Step 1: Factorize original categorical columns
    for col in CAT_COLS:
        codes, uniques = train[col].factorize()
        category_map[col] = uniques
        train[col] = pd.Categorical.from_codes(codes, uniques)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        tst_codes = test[col].map(code_map).fillna(-1).astype('int32')
        test[col] = pd.Categorical.from_codes(tst_codes, uniques)

    # Step 2: Num2cat (RealMLP-style: factorize + direct int32→category)
    for col in NUM_COLS:
        cat_name = f'{col}_cat_'
        if col == 'calorie_expenditure':
            # Direct int32→category (no factorize)
            train[cat_name] = (train[col] // 5).astype('int32')
            test[cat_name] = (test[col] // 5).astype('int32')
            # Factorize for consistency
            codes, uniques = train[cat_name].factorize()
            category_map[cat_name] = uniques
            train[cat_name] = pd.Categorical.from_codes(codes, uniques)
            code_map = {cat: i for i, cat in enumerate(uniques)}
            tst_codes = test[cat_name].map(code_map).fillna(-1).astype('int32')
            test[cat_name] = pd.Categorical.from_codes(tst_codes, uniques)
            all_cat_cols.append(cat_name)
            continue

        if col == 'water_intake':
            wname = 'water_intake_cat2_'
            if water_round:
                train[wname] = (train[col] * 50).round().astype('int32')
                test[wname] = (test[col] * 50).round().astype('int32')
            else:
                train[wname] = (train[col] * 50).astype('int32')
                test[wname] = (test[col] * 50).astype('int32')
            codes, uniques = train[wname].factorize()
            category_map[wname] = uniques
            train[wname] = pd.Categorical.from_codes(codes, uniques)
            code_map = {cat: i for i, cat in enumerate(uniques)}
            tst_codes = test[wname].map(code_map).fillna(-1).astype('int32')
            test[wname] = pd.Categorical.from_codes(tst_codes, uniques)
            all_cat_cols.append(wname)

        if extra_cat_flag:
            if col == 'sleep_duration':
                cname = 'sleep_duration_cat2_'
                train[cname] = (train[col] * 10).round().astype('int32')
                test[cname] = (test[col] * 10).round().astype('int32')
                codes, uniques = train[cname].factorize()
                category_map[cname] = uniques
                train[cname] = pd.Categorical.from_codes(codes, uniques)
                test[cname] = pd.Categorical.from_codes(
                    test[cname].map({cat: i for i, cat in enumerate(uniques)}).fillna(-1).astype('int32'), uniques)
                all_cat_cols.append(cname)
            if col == 'bmi':
                for cname, expr in [('bmi_cat2_', lambda x: (x - 18.5).round().astype('int32')),
                                     ('bmi_cat1_', lambda x: (24.9 - x).round().astype('int32'))]:
                    train[cname] = expr(train[col])
                    test[cname] = expr(test[col])
                    codes, uniques = train[cname].factorize()
                    category_map[cname] = uniques
                    train[cname] = pd.Categorical.from_codes(codes, uniques)
                    test[cname] = pd.Categorical.from_codes(
                        test[cname].map({cat: i for i, cat in enumerate(uniques)}).fillna(-1).astype('int32'), uniques)
                    all_cat_cols.append(cname)
            if col == 'heart_rate':
                for cname, expr in [('heart_rate_cat2_lo', lambda x: (x - 60).round().astype('int32')),
                                     ('heart_rate_cat2_hi', lambda x: (100 - x).round().astype('int32'))]:
                    train[cname] = expr(train[col])
                    test[cname] = expr(test[col])
                    codes, uniques = train[cname].factorize()
                    category_map[cname] = uniques
                    train[cname] = pd.Categorical.from_codes(codes, uniques)
                    test[cname] = pd.Categorical.from_codes(
                        test[cname].map({cat: i for i, cat in enumerate(uniques)}).fillna(-1).astype('int32'), uniques)
                    all_cat_cols.append(cname)

        # Generic {col}_cat_ via factorize
        round_flag = (col == 'step_count')
        if round_flag:
            tr_series = train[col].round(-1)
            tst_series = test[col].round(-1)
        else:
            tr_series = train[col]
            tst_series = test[col]
        codes, uniques = tr_series.factorize()
        cat_map_key = f'__num2cat__{col}'
        category_map[cat_map_key] = {'uniques': uniques, 'round_flag': round_flag}
        train[cat_name] = pd.Categorical.from_codes(codes, uniques)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        tst_codes = tst_series.map(code_map).fillna(-1).astype('int32')
        test[cat_name] = pd.Categorical.from_codes(tst_codes, uniques)
        all_cat_cols.append(cat_name)

    # Step 3: KBins discretization (sleep_duration 70bins, water_intake 10bins)
    bin_config = {'sleep_duration': [70], 'water_intake': [10]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f'{col}_{n_bins}_{strategy}_bin_'
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                       strategy=strategy, subsample=None)
                binned_tr = kb.fit_transform(train[[col]]).ravel().astype('int32')
                binned_tst = kb.transform(test[[col]]).ravel().astype('int32')
                codes, uniques = pd.Series(binned_tr).factorize()
                category_map[bin_name] = uniques
                train[bin_name] = pd.Categorical.from_codes(codes, uniques)
                code_map = {cat: i for i, cat in enumerate(uniques)}
                tst_codes = pd.Series(binned_tst).map(code_map).fillna(-1).astype('int32')
                test[bin_name] = pd.Categorical.from_codes(tst_codes, uniques)
                all_cat_cols.append(bin_name)

    # Step 4: Interaction categories
    important_combos = [('heart_rate', 'bmi')]
    if stress_pal_flag:
        important_combos.append(('stress_level', 'physical_activity_level'))
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        tr_combo = train[cols[0]].astype(str)
        for c in cols[1:]:
            tr_combo = tr_combo + '_' + train[c].astype(str)
        tst_combo = test[cols[0]].astype(str)
        for c in cols[1:]:
            tst_combo = tst_combo + '_' + test[c].astype(str)
        codes, uniques = pd.factorize(tr_combo, sort=False)
        category_map[combo_name] = uniques
        train[combo_name] = pd.Categorical.from_codes(codes, uniques)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        tst_codes = tst_combo.map(code_map).fillna(-1).astype('int32')
        test[combo_name] = pd.Categorical.from_codes(tst_codes, uniques)
        all_cat_cols.append(combo_name)

    # Step 5: GP nonlinear features (numeric)
    if gp_flag:
        add_gp_features(train, test)

    # Step 6: Sleep features (optional)
    if fe_config.get('sleep_cat', False):
        add_sleep_cat(train, test)
        codes, uniques = train['sleep_cat_'].factorize()
        category_map['sleep_cat_'] = uniques
        train['sleep_cat_'] = pd.Categorical.from_codes(codes, uniques)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        tst_codes = test['sleep_cat_'].map(code_map).fillna(-1).astype('int32')
        test['sleep_cat_'] = pd.Categorical.from_codes(tst_codes, uniques)
        all_cat_cols.append('sleep_cat_')

        sleep_interact_cols = fe_config.get('sleep_interact_with', None)
        if sleep_interact_cols:
            # Resolve 'stress_pal' → 'stress_level_physical_activity_level_'
            resolved = []
            for c in sleep_interact_cols:
                if c in ('stress_pal', 'stress_level_physical_activity_level_'):
                    resolved.append('stress_level_physical_activity_level_')
                else:
                    resolved.append(c)
            new_cols = add_sleep_interact(train, test, resolved)
            for col in new_cols:
                codes, uniques = train[col].factorize()
                category_map[col] = uniques
                train[col] = pd.Categorical.from_codes(codes, uniques)
                code_map = {cat: i for i, cat in enumerate(uniques)}
                tst_codes = test[col].map(code_map).fillna(-1).astype('int32')
                test[col] = pd.Categorical.from_codes(tst_codes, uniques)
                all_cat_cols.append(col)

    # Step 7: Collect feature names
    new_cat_cols = [col for col in train.columns if col.endswith('_') and col not in CAT_COLS]
    new_num_cols = [col for col in train.columns if col.startswith('_gp_')]
    cat_col_names = sorted(CAT_COLS + new_cat_cols)
    num_col_names = sorted([c for c in NUM_COLS] + new_num_cols)
    feature_names = sorted(cat_col_names + num_col_names)

    # Sort columns for consistency
    X_train = train.reindex(feature_names, axis=1)
    X_test = test.reindex(feature_names, axis=1)

    return X_train, X_test, cat_col_names, feature_names, category_map


def get_or_create_features(fe_config, y=None):
    """Config-driven FE with caching.

    Args:
        fe_config: dict with keys:
            - num2cat: bool (A_num2cat)
            - kbins: bool
            - heart_bmi: bool
            - te: bool
            - num_fill: 'median' or 'zero'
            - te_folds: int (default 5)
            - format: 'gbdt' (default) or 'realmlp'
            - gp_features: bool (default False, for 'realmlp' format)
            - extra_cat_features: bool (default False, for 'realmlp' format)
            - water_intake_round: bool (default True, for 'realmlp' format)
        y: target array (needed for TE; can be None if cache exists)

    Returns:
        X_train, X_test, cat_info, feature_names, tag
        where cat_info is:
          - format='gbdt': list of int indices (cat_indices)
          - format='realmlp': list of str column names (cat_col_names)
    """
    fmt = fe_config.get('format', 'gbdt')
    tag = build_tag(fe_config)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache
    meta_path = os.path.join(CACHE_DIR, f'{tag}_meta.json')
    train_path = os.path.join(CACHE_DIR, f'{tag}_train.pkl')
    test_path = os.path.join(CACHE_DIR, f'{tag}_test.pkl')

    if os.path.exists(meta_path) and os.path.exists(train_path) and os.path.exists(test_path):
        print(f'[FE] Cache hit: {tag}', flush=True)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        X_train = pd.read_pickle(train_path)
        X_test = pd.read_pickle(test_path)
        cat_info = meta.get('cat_col_names', meta.get('cat_indices', []))
        return X_train, X_test, cat_info, meta['feature_names'], tag

    # Cache miss — run FE
    print(f'[FE] Cache miss: {tag}, running feature engineering...', flush=True)

    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    train = train.copy()
    test = test.copy()

    # --- RealMLP path ---
    if fmt == 'realmlp':
        X_train, X_test, cat_col_names, feature_names, _ = _realmlp_fe_transform(
            train, test, fe_config)

        # Save to cache
        X_train.to_pickle(train_path)
        X_test.to_pickle(test_path)
        meta = {
            'tag': tag,
            'cat_col_names': cat_col_names,
            'feature_names': feature_names,
            'ndim': len(feature_names),
            'config': fe_config,
            'format': 'realmlp',
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'[FE] Saved cache: {tag} ({len(feature_names)}d, format=realmlp)', flush=True)
        return X_train, X_test, cat_col_names, feature_names, tag

    # --- GBDT path (original behavior) ---
    if y is None:
        raise ValueError('y (target) is required when FE cache miss (needed for TE)')

    extra_cat = []
    extra_num = []

    # Step 0: Ordinal encode categoricals BEFORE fill_missing (NaN preserved)
    # Must happen before num2cat/stress_pal which need string values,
    # so only apply to base CAT_COLS, not derived cat columns.
    ordinal_done = False
    if fe_config.get('ordinal_encode', False):
        ordinal_encode(train, test, CAT_COLS)
        ordinal_done = True

    # Step 1: num2cat BEFORE fill missing (NaN -> 'missing' category)
    # Skip if ordinal_encode=True (we don't mix ordinal with num2cat/stress_pal)
    if fe_config.get('num2cat', False):
        extra_cat.extend(add_num2cat(train, test))

    # Step 2: stress_pal BEFORE fill missing (optional)
    if fe_config.get('stress_pal', True) and not ordinal_done:
        extra_cat.extend(add_stress_pal(train, test))

    # Step 2b: sleep_cat BEFORE fill missing (NaN -> 'missing' category)
    if fe_config.get('sleep_cat', False):
        extra_cat.extend(add_sleep_cat(train, test))

    # Step 2b2: sleep_interact BEFORE fill missing (requires sleep_cat first)
    sleep_interact_cols = fe_config.get('sleep_interact_with', None)
    if sleep_interact_cols:
        extra_cat.extend(add_sleep_interact(train, test, sleep_interact_cols))

    # Step 2c: stress_bin BEFORE fill missing (NaN stress -> 0)
    if fe_config.get('stress_bin', False):
        extra_num.extend(add_stress_bin(train, test))

    # Step 2d: key_rule3 BEFORE fill missing (requires sleep_cat already created)
    if fe_config.get('key_rule3', False):
        extra_cat.extend(add_key_rule3(train, test))

    # Step 3: Fill missing
    # If ordinal_encode=True, preserve all NaN (reference notebook: models handle NaN natively)
    # Domain features are computed AFTER encoding with NaN preserved (ratios produce NaN/inf→NaN)
    if ordinal_done:
        # Don't fill any NaN — reference notebook lets LGBM/XGB handle NaN natively
        pass
    else:
        fill_missing(train, test, num_strategy=fe_config.get('num_fill', 'median'))

    # Step 4: kbins (after fill, needs no NaN)
    if fe_config.get('kbins', False):
        extra_cat.extend(add_kbins(train, test))

    # Step 5: heart_bmi (after fill, qcut needs no NaN)
    if fe_config.get('heart_bmi', False):
        extra_cat.extend(add_heart_bmi(train, test))

    # All cat columns
    all_cat = CAT_COLS + extra_cat

    # Step 6: TE (before label_encode, needs string values)
    te_names = []
    if fe_config.get('te', False):
        te_folds = fe_config.get('te_folds', 5)
        te_cols = [c for c in all_cat if not c.endswith('bin_')]
        te_names = target_encode(train, test, y, te_cols, n_splits=te_folds,
                                 random_state=fe_config.get('te_random_state', None))

    # Step 6b: Num TE (TE on numeric columns as strings)
    num_te_names = []
    if fe_config.get('num_te', False):
        te_folds = fe_config.get('te_folds', 5)
        num_te_names = target_encode_numeric(train, test, y, n_splits=te_folds,
                                             random_state=fe_config.get('te_random_state', None))

    # Step 7: Encode categoricals
    if ordinal_done:
        # Ordinal-encoded columns already done (Step 0)
        # Only label-encode the extra cat columns (num2cat, stress_pal, etc.)
        if extra_cat:
            label_encode(train, test, extra_cat)
    else:
        label_encode(train, test, all_cat)

    # Step 7b: Domain features (after encoding, uses ordinal values)
    domain_names = []
    if fe_config.get('domain_features', False):
        variant = fe_config.get('domain_features_variant', 'full')
        replace_inf = fe_config.get('replace_inf', True)  # False=keep inf (ref notebook), True=inf→NaN (XGB)
        domain_names = add_domain_features(train, test, variant=variant, replace_inf=replace_inf)
        extra_num.extend(domain_names)

    # Build feature matrix
    # Column order: default is NUM + extra_num + cat + TE (pipeline convention)
    # Set col_order='cat_first' for NUM + cat + extra_num + TE (reference notebook convention)
    col_order = fe_config.get('col_order', 'default')
    if col_order == 'cat_first':
        feature_cols = NUM_COLS + all_cat + extra_num + te_names + num_te_names
    else:
        feature_cols = NUM_COLS + extra_num + all_cat + te_names + num_te_names

    # Drop specified features (post-FE column removal)
    drop = fe_config.get('drop_features', [])
    if drop:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in drop]
        # Recompute cat_indices after drop
        dropped = before - len(feature_cols)
        print(f'[FE] Dropped {dropped} features: {drop}', flush=True)

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    # Compute cat_indices from actual column positions (works for any col_order)
    cat_set = set(all_cat) - set(drop) if drop else set(all_cat)
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_set]

    # Save to cache (pickle — no pyarrow dependency)
    X_train.to_pickle(train_path)
    X_test.to_pickle(test_path)
    meta = {
        'tag': tag,
        'cat_indices': cat_indices,
        'feature_names': feature_cols,
        'ndim': len(feature_cols),
        'config': fe_config,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[FE] Saved cache: {tag} ({len(feature_cols)}d)', flush=True)
    return X_train, X_test, cat_indices, feature_cols, tag
