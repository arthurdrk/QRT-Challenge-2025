from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import typer
from rich.console import Console as RichConsole
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

try:
    import optuna
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    optuna = None
    XGBRegressor = None

from qrtdc.cli._console import Console

app = typer.Typer(help="Preprocessing pipeline that mirrors the exploration notebook.")


@dataclass
class Paths:
    clinical_train: Path
    clinical_eval: Path
    molecular_train: Path
    molecular_eval: Path
    target_train: Path
    out_train: Path
    out_eval: Path


NUMERIC_COLS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
CYTO_COL = "CYTOGENETICS"
MUTATION_FEATURES = [
    "Nmut", "VAF_avg", "VAF_std", "VAF_max", "LEN_avg", "LEN_std", "LEN_max"
]

# --- Cytogenetics regex machinery (copied from the notebook) ---
_ISCN_EVENT_RE = re.compile(r"(del|dup|inv|ins|i|t|add|der)\s*\(", re.IGNORECASE)
_MONOSOMY_RE = re.compile(r"(?<![pq])-(\d{1,2}|X|Y)(?![pq])", re.IGNORECASE)
_TRISOMY_RE = re.compile(r"(?<![pq])\+(\d{1,2}|X|Y)(?![pq])", re.IGNORECASE)
_CHR_NUM_RE = re.compile(r"(?<![pq])(\d{1,2}|X|Y)(?![pq])", re.IGNORECASE)

_MINUS5_OR_DEL5Q_RE = re.compile(r"-(?:5)(?![pq])|del\s*\(\s*5\s*\)\s*\(\s*q", re.IGNORECASE)
_MINUS7_OR_DEL7Q_RE = re.compile(r"-(?:7)(?![pq])|del\s*\(\s*7\s*\)\s*\(\s*q", re.IGNORECASE)
_PLUS8_RE = re.compile(r"\+8(?![pq])", re.IGNORECASE)
_T_8_21_RE = re.compile(r"t\s*\(\s*8\s*;\s*21\s*\)", re.IGNORECASE)
_INV16_OR_T_16_16_RE = re.compile(r"(inv\s*\(\s*16\s*\)|t\s*\(\s*16\s*;\s*16\s*\))", re.IGNORECASE)
_T_15_17_RE = re.compile(r"t\s*\(\s*15\s*;\s*17\s*\)", re.IGNORECASE)
_STRUCTURAL_RE = re.compile(r"(del|dup|inv|ins|i|t|add|der)\s*\(", re.IGNORECASE)


def _split_clones(karyo: str) -> List[str]:
    return [c.strip() for c in str(karyo).split('/') if c.strip()]


def _extract_metaphases(clone: str) -> int:
    m = re.search(r"\[(\d+)\]", clone)
    return int(m.group(1)) if m else 0


def _count_events(clone: str) -> int:
    n_struct = len(_ISCN_EVENT_RE.findall(clone))
    n_mono = len(_MONOSOMY_RE.findall(clone))
    n_tri = len(_TRISOMY_RE.findall(clone))
    n_mono_minusY = len(re.findall(r"(?<![pq])-(?:Y)(?![pq])", clone, flags=re.IGNORECASE))
    return n_struct + n_tri + max(n_mono - n_mono_minusY, 0)


def _chromosomes_altered(clone: str) -> int:
    nums = set()
    for m in _MONOSOMY_RE.finditer(clone):
        nums.add(m.group(1).upper())
    for m in _TRISOMY_RE.finditer(clone):
        nums.add(m.group(1).upper())
    for ev in re.finditer(r"(del|dup|inv|ins|i|t|add|der)\s*\(([^)]+)\)", clone, flags=re.IGNORECASE):
        for x in re.split(r"[;,\s]+", ev.group(2)):
            if _CHR_NUM_RE.fullmatch(x.strip()):
                nums.add(x.strip().upper())
    nums.discard('Y')
    return len(nums)


def _has_structural(clone: str) -> bool:
    return bool(_STRUCTURAL_RE.search(clone))


def _autosomic_monosomies(clone: str) -> List[int]:
    return [int(m.group(1)) for m in _MONOSOMY_RE.finditer(clone) if m.group(1).upper() not in ('X', 'Y')]


def _is_monosomal_karyotype(karyo: str) -> bool:
    clones = _split_clones(karyo)
    autosomal_monosomies = set()
    any_struct = False
    for c in clones:
        autosomal_monosomies.update(_autosomic_monosomies(c))
        any_struct = any_struct or _has_structural(c)
    return (len(autosomal_monosomies) >= 2) or (len(autosomal_monosomies) >= 1 and any_struct)


def _is_complex_karyotype(karyo: str) -> bool:
    clones = _split_clones(karyo)
    total_events = 0
    for c in clones:
        c_wo_minusY = re.sub(r"(?<![pq])-(?:Y)(?![pq])", '', c, flags=re.IGNORECASE)
        total_events += _count_events(c_wo_minusY)
    return total_events >= 3


def _clone_flags(clone: str) -> Dict[str, bool]:
    return {
        'minus5_or_del5q': bool(_MINUS5_OR_DEL5Q_RE.search(clone)),
        'minus7_or_del7q': bool(_MINUS7_OR_DEL7Q_RE.search(clone)),
        'plus8': bool(_PLUS8_RE.search(clone)),
        't_8_21': bool(_T_8_21_RE.search(clone)),
        'inv16_or_t_16_16': bool(_INV16_OR_T_16_16_RE.search(clone)),
        't_15_17': bool(_T_15_17_RE.search(clone)),
        'has_structural': _has_structural(clone),
        'events_count': _count_events(clone),
        'chrs_altered': _chromosomes_altered(clone),
        'has_any_abn': bool(_ISCN_EVENT_RE.search(clone) or _MONOSOMY_RE.search(clone) or _TRISOMY_RE.search(clone)),
    }


def add_cytogenetics_features(df: pd.DataFrame, col: str = CYTO_COL) -> pd.DataFrame:
    rows = []
    for k in df[col]:
        clones = _split_clones(k)
        clone_info = []
        total_meta_known = 0
        for c in clones:
            n_meta = _extract_metaphases(c)
            flags = _clone_flags(c)
            clone_info.append((c, n_meta, flags))
            total_meta_known += n_meta

        any_abn = any(f['has_any_abn'] for _, _, f in clone_info)
        n_events = sum(f['events_count'] for _, _, f in clone_info)
        n_chrs = max([f['chrs_altered'] for _, _, f in clone_info] + [0])

        has_minus5_or_del5q = any(f['minus5_or_del5q'] for _, _, f in clone_info)
        has_minus7_or_del7q = any(f['minus7_or_del7q'] for _, _, f in clone_info)
        has_plus8 = any(f['plus8'] for _, _, f in clone_info)
        has_t_8_21 = any(f['t_8_21'] for _, _, f in clone_info)
        has_inv16_or_t_16_16 = any(f['inv16_or_t_16_16'] for _, _, f in clone_info)
        has_t_15_17 = any(f['t_15_17'] for _, _, f in clone_info)

        is_mk = _is_monosomal_karyotype(k)
        is_ck = _is_complex_karyotype(k)
        eln_like_adverse = bool(is_mk or is_ck or has_minus5_or_del5q or has_minus7_or_del7q)

        def _prop(cond_fn):
            if total_meta_known == 0:
                return 0.0
            pos = sum(n_meta for _, n_meta, f in clone_info if n_meta and cond_fn(f))
            return pos / total_meta_known if total_meta_known else 0.0

        rows.append({
            'has_any_abnormality': int(any_abn),
            'n_events': int(n_events),
            'n_chromosomes_altered': int(n_chrs),
            'has_minus5_or_del5q': int(has_minus5_or_del5q),
            'has_minus7_or_del7q': int(has_minus7_or_del7q),
            'has_plus8': int(has_plus8),
            'has_t_8_21': int(has_t_8_21),
            'has_inv16_or_t_16_16': int(has_inv16_or_t_16_16),
            'has_t_15_17': int(has_t_15_17),
            'is_monosomal_karyotype': int(is_mk),
            'is_complex_karyotype': int(is_ck),
            'eln_like_flag_adverse_cyto': int(eln_like_adverse),
            'total_metaphases': int(total_meta_known),
            'prop_any_abnormal': float(_prop(lambda f: f['has_any_abn'])),
            'prop_adverse_5_7': float(_prop(lambda f: f['minus5_or_del5q'] or f['minus7_or_del7q'])),
            'prop_plus8': float(_prop(lambda f: f['plus8'])),
            'prop_favorable_core': float(_prop(lambda f: f['t_8_21'] or f['inv16_or_t_16_16'])),
        })
    features_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.copy(), features_df], axis=1).drop(columns=[col])


# --- Mutation feature builder ---

def compute_mutation_features(maf_df: pd.DataFrame, X_df: pd.DataFrame) -> pd.DataFrame:
    maf_df = maf_df.copy()
    maf_df['LEN'] = maf_df['END'] - maf_df['START'] + 1
    tmp = maf_df.groupby('ID').agg(
        Nmut=('ID', 'size'),
        VAF_avg=('VAF', 'mean'),
        VAF_std=('VAF', 'std'),
        VAF_max=('VAF', 'max'),
        LEN_avg=('LEN', 'mean'),
        LEN_std=('LEN', 'std'),
        LEN_max=('LEN', 'max')
    ).reset_index()
    tmp[['VAF_std', 'LEN_std']] = tmp[['VAF_std', 'LEN_std']].fillna(0)
    X_w_mut = X_df.merge(tmp, on='ID', how='left').fillna({
        'Nmut': 0, 'VAF_avg': 0, 'VAF_std': 0, 'VAF_max': 0,
        'LEN_avg': 0, 'LEN_std': 0, 'LEN_max': 0
    })
    return X_w_mut


# --- Utilities ---

def _validate_dependencies():
    if optuna is None or XGBRegressor is None:
        Console.print("[error]This command requires optional dependencies: optuna, xgboost[/error]")
        raise typer.Exit(code=2)


def _fit_imputers(X: pd.DataFrame, numeric_cols: List[str], seed: int, n_trials: int, n_jobs: int) -> Dict[str, XGBRegressor]:
    Console.print(f"[title]Imputation[/title] :: Searching XGBoost models for columns with missing values...")
    missing_cols = [c for c in numeric_cols if X[c].isnull().any()]
    if not missing_cols:
        Console.print("[success]No missing numeric columns in training after row filter — skipping imputation[/success]")
        return {}

    imputers: Dict[str, XGBRegressor] = {}

    for col in missing_cols:
        Console.print(f"[info]Training XGBoost imputer for [bold]{col}[/bold] on rows where it is not missing[/info]")
        mask = X[col].notnull()
        X_train = X.loc[mask, numeric_cols].drop(columns=[col])
        y_train = X.loc[mask, col]
        X_train = X_train.fillna(X_train.median())

        def objective(trial: 'optuna.trial.Trial') -> float:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': seed,
                'objective': 'reg:squarederror',
                'n_jobs': n_jobs,
            }
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=n_jobs)
            return -scores.mean()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), transient=True) as progress:
            task = progress.add_task(f"Optimizing {col}", total=None)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            progress.update(task, completed=True)

        Console.print(f"[success]Best params for {col}[/success]: {json.dumps(study.best_params)}")
        model = XGBRegressor(**{**study.best_params, 'random_state': seed, 'objective': 'reg:squarederror', 'n_jobs': n_jobs})
        model.fit(X_train, y_train)
        imputers[col] = model

    return imputers


def _apply_imputers(df: pd.DataFrame, imputers: Dict[str, XGBRegressor], numeric_cols: List[str], dataset_name: str) -> pd.DataFrame:
    if not imputers:
        return df
    for col, model in imputers.items():
        if df[col].isnull().sum() == 0:
            continue
        Console.print(f"[info]Imputing missing values for [bold]{col}[/bold] in {dataset_name}")
        mask = df[col].isnull()
        X_pred = df.loc[mask, numeric_cols].drop(columns=[col])
        X_pred = X_pred.fillna(df[numeric_cols].drop(columns=[col]).median())
        df.loc[mask, col] = model.predict(X_pred)
        Console.print(f"[success]Imputed {int(mask.sum())} values in {dataset_name}[/success]")
    nulls = df[numeric_cols].isnull().sum().to_dict()
    Console.print(f"[info]Missing-values summary in {dataset_name} after imputation: {nulls}")
    return df


@app.command()
def run(
    clinical_train: Path = typer.Option(Path("data/clinical_train.csv"), help="Path to clinical_train.csv"),
    clinical_eval: Path = typer.Option(Path("data/clinical_val.csv"), help="Path to clinical_val.csv"),
    molecular_train: Path = typer.Option(Path("data/molecular_train.csv"), help="Path to molecular_train.csv"),
    molecular_eval: Path = typer.Option(Path("data/molecular_val.csv"), help="Path to molecular_val.csv"),
    target_train: Path = typer.Option(Path("data/target_train.csv"), help="Path to target_train.csv"),
    out_train: Path = typer.Option(Path("data/train_enhanced.csv"), help="Output CSV for enhanced training set"),
    out_eval: Path = typer.Option(Path("data/eval_enhanced.csv"), help="Output CSV for enhanced eval set"),
    numeric_drop_threshold: int = typer.Option(4, help="Drop training rows with more than this many missing numeric values"),
    cytogen_const_threshold: float = typer.Option(0.95, help="Threshold to drop nearly-constant cytogenetic features (proportion)"),
    seed: int = typer.Option(42, help="Random seed"),
    n_trials: int = typer.Option(250, help="Optuna trials per imputed column"),
    n_jobs: int = typer.Option(-1, help="Parallel jobs for XGBoost and CV"),
) -> None:
    """Run the preprocessing pipeline and write enhanced CSVs.

    This function mirrors the exploration notebook while emitting detailed, English logs.
    """
    _validate_dependencies()

    Console.print("[title]Preprocessing[/title] :: Starting")

    # 1) Load CSVs
    Console.print("[info]Loading input CSV files")
    X_train = pd.read_csv(clinical_train)
    X_eval = pd.read_csv(clinical_eval)
    maf_train = pd.read_csv(molecular_train)
    maf_eval = pd.read_csv(molecular_eval)
    target = pd.read_csv(target_train)

    # 2) Target cleanup and alignment
    target_cols = ['OS_YEARS', 'OS_STATUS']
    before = len(target)
    target = target.dropna(subset=target_cols).copy()
    target['OS_YEARS'] = pd.to_numeric(target['OS_YEARS'], errors='coerce')
    target['OS_STATUS'] = target['OS_STATUS'].astype(bool)
    Console.print(f"[info]Target cleaned: {before} -> {len(target)} rows; columns: {target_cols}")

    features = ['ID', *NUMERIC_COLS, CYTO_COL]
    X_train = X_train.loc[X_train['ID'].isin(target['ID']), features].copy()
    X_eval = X_eval.loc[:, features].copy()

    # 3) Row filter on missing counts (train only)
    num_cols = NUMERIC_COLS
    num_missing = X_train[num_cols].isna().sum(axis=1)
    to_drop = num_missing > numeric_drop_threshold
    dropped = int(to_drop.sum())
    X_train = X_train.loc[~to_drop].copy()
    Console.print(f"[info]Dropping training rows with > {numeric_drop_threshold} missing numeric values: {dropped} dropped, {len(X_train)} remain")

    # 4) Robust scale numeric clinical features
    num_scaler = RobustScaler()
    Console.print(f"[info]Fitting RobustScaler on numeric clinical features: {num_cols}")
    X_train[num_cols] = num_scaler.fit_transform(X_train[num_cols])
    X_eval[num_cols] = num_scaler.transform(X_eval[num_cols])

    # 5) XGBoost imputers per missing column (train fit) and apply to train/eval
    imputers = _fit_imputers(X_train, num_cols, seed=seed, n_trials=n_trials, n_jobs=n_jobs)
    X_train = _apply_imputers(X_train, imputers, num_cols, dataset_name="train")
    X_eval = _apply_imputers(X_eval, imputers, num_cols, dataset_name="eval")

    # 6) Mutation features + scaling
    Console.print("[title]Mutation features[/title] :: Building and scaling")
    X_train_mut = compute_mutation_features(maf_train, X_train)
    X_eval_mut = compute_mutation_features(maf_eval, X_eval)

    mut_scaler = RobustScaler()
    Console.print(f"[info]Fitting RobustScaler on mutation features: {MUTATION_FEATURES}")
    X_train_mut[MUTATION_FEATURES] = mut_scaler.fit_transform(X_train_mut[MUTATION_FEATURES])
    X_eval_mut[MUTATION_FEATURES] = mut_scaler.transform(X_eval_mut[MUTATION_FEATURES])

    # 7) Cytogenetics features
    Console.print("[title]Cytogenetics[/title] :: Parsing ISCN strings to engineered features")
    X_train_cyto = add_cytogenetics_features(X_train_mut)
    X_eval_cyto = add_cytogenetics_features(X_eval_mut)

    cytogenetics_features = [
        'has_any_abnormality', 'n_events', 'n_chromosomes_altered',
        'has_minus5_or_del5q', 'has_minus7_or_del7q', 'has_plus8',
        'has_t_8_21', 'has_inv16_or_t_16_16', 'has_t_15_17',
        'is_monosomal_karyotype', 'is_complex_karyotype',
        'eln_like_flag_adverse_cyto', 'total_metaphases',
        'prop_any_abnormal', 'prop_adverse_5_7', 'prop_plus8',
        'prop_favorable_core'
    ]

    # 7a) Drop nearly-constant features on training
    Console.print(f"[info]Detecting nearly-constant cytogenetics features (≥ {cytogen_const_threshold:.0%} same value)")
    nearly_constant: List[str] = []
    for col in cytogenetics_features:
        vc = X_train_cyto[col].value_counts()
        if len(vc) == 0:
            continue
        max_prop = float(vc.iloc[0]) / float(len(X_train_cyto)) if len(X_train_cyto) else 0.0
        if max_prop >= cytogen_const_threshold:
            nearly_constant.append(col)
            RichConsole().print(f"Removing {col}: {max_prop:.2%} of values are {vc.index[0]}")

    if nearly_constant:
        Console.print(f"[warning]Dropping {len(nearly_constant)} nearly-constant features: {nearly_constant}[/warning]")
        X_train_cyto = X_train_cyto.drop(columns=nearly_constant)
        X_eval_cyto = X_eval_cyto.drop(columns=nearly_constant)
    else:
        Console.print("[success]No nearly-constant cytogenetics features detected[/success]")

    kept_cyto = [f for f in cytogenetics_features if f not in nearly_constant]

    # 7b) Robust-scale remaining cytogenetics features
    if kept_cyto:
        cyto_scaler = RobustScaler()
        Console.print(f"[info]Fitting RobustScaler on cytogenetics features: {kept_cyto}")
        X_train_cyto[kept_cyto] = cyto_scaler.fit_transform(X_train_cyto[kept_cyto])
        X_eval_cyto[kept_cyto] = cyto_scaler.transform(X_eval_cyto[kept_cyto])
    else:
        Console.print("[warning]No cytogenetics features left to scale after constant-feature removal[/warning]")

    # 8) Merge target into train and write outputs
    Console.print("[title]Output[/title] :: Writing enhanced CSVs")
    df_train_out = X_train_cyto.merge(target, on='ID', how='left')
    df_eval_out = X_eval_cyto

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    df_train_out.to_csv(out_train, index=False)
    df_eval_out.to_csv(out_eval, index=False)

    Console.print(f"[success]Wrote training to: {out_train}[/success]")
    Console.print(f"[success]Wrote evaluation to: {out_eval}[/success]")
    Console.print("[title]Preprocessing[/title] :: Completed")
