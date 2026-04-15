"""
Script d'entrainement reproductible du modele ML NB10.

Charge le dataset EM-DAT multi-type, construit les features (lags par pays x type
+ warming NASA + cross-type), entraine le meilleur modele de regression et
classification, puis sauvegarde les artefacts dans outputs/.

Appele par le CI/CD avant le build Docker pour generer les artefacts sans
dependre du notebook. Equivalent au NB10 executable mais en format .py.

Usage :
    python scripts/train_nb10.py
"""

import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

try:
    import mlflow
    from mlflow.models.signature import infer_signature
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def _git_commit() -> str:
    """Retourne le SHA court du commit courant, ou 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.getenv("GITHUB_SHA", "unknown")[:7]


def _git_branch() -> str:
    """Retourne la branche courante."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.getenv("GITHUB_REF_NAME", "unknown")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "decadal-deaths-disasters-by-type.csv"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CLIMATE_TYPES = ["drought", "flood", "extreme_weather", "extreme_temperature", "wildfire"]

AGGREGATES = [
    "World", "Asia", "Europe", "Africa", "North America", "South America",
    "Oceania", "USSR", "European Union (27)",
    "High-income countries", "Low-income countries",
    "Lower-middle-income countries", "Upper-middle-income countries",
]

# Source : NASA GISS GISTEMP v4, baseline 1951-1980. Valeur 2030 = IPCC SSP2-4.5.
WARMING_INDEX = {
    1900: -0.17, 1910: -0.30, 1920: -0.20, 1930: -0.10, 1940: 0.04,
    1950: -0.03, 1960: -0.02, 1970: 0.00, 1980: 0.20, 1990: 0.35,
    2000: 0.59, 2010: 0.81, 2020: 1.00, 2030: 1.40,
}

RISK_NAMES = ["Aucun", "Faible", "Modere", "Eleve", "Critique"]
RISK_MAP = {name: i for i, name in enumerate(RISK_NAMES)}

FEATURES_REG = [
    "decade_index", "continent_enc", "type_enc",
    "log_lag_1", "log_lag_2", "trend",
    "log_cumul_mean", "cumul_std", "cumul_max",
    "log_impact_other", "warming", "warming_sq",
]

# Mapping continent (meme que NB10)
CONTINENT_MAP = {
    "Afghanistan": "Asie", "Albania": "Europe", "Algeria": "Afrique",
    "Angola": "Afrique", "Argentina": "Amerique du Sud", "Armenia": "Asie",
    "Australia": "Oceanie", "Austria": "Europe", "Azerbaijan": "Asie",
    "Bangladesh": "Asie", "Belarus": "Europe", "Belgium": "Europe",
    "Benin": "Afrique", "Bhutan": "Asie", "Bolivia": "Amerique du Sud",
    "Bosnia and Herzegovina": "Europe", "Botswana": "Afrique",
    "Brazil": "Amerique du Sud", "Bulgaria": "Europe", "Burkina Faso": "Afrique",
    "Burundi": "Afrique", "Cambodia": "Asie", "Cameroon": "Afrique",
    "Canada": "Amerique du Nord", "Central African Republic": "Afrique",
    "Chad": "Afrique", "Chile": "Amerique du Sud", "China": "Asie",
    "Colombia": "Amerique du Sud", "Comoros": "Afrique", "Congo": "Afrique",
    "Costa Rica": "Amerique centrale", "Croatia": "Europe", "Cuba": "Caraibes",
    "Cyprus": "Europe", "Czechia": "Europe", "Denmark": "Europe",
    "Democratic Republic of Congo": "Afrique", "Djibouti": "Afrique",
    "Dominican Republic": "Caraibes", "East Timor": "Asie",
    "Ecuador": "Amerique du Sud", "Egypt": "Afrique",
    "El Salvador": "Amerique centrale", "Eritrea": "Afrique", "Estonia": "Europe",
    "Eswatini": "Afrique", "Ethiopia": "Afrique", "Fiji": "Oceanie",
    "Finland": "Europe", "France": "Europe", "Gabon": "Afrique",
    "Gambia": "Afrique", "Georgia": "Asie", "Germany": "Europe",
    "Ghana": "Afrique", "Greece": "Europe", "Guatemala": "Amerique centrale",
    "Guinea": "Afrique", "Guinea-Bissau": "Afrique", "Guyana": "Amerique du Sud",
    "Haiti": "Caraibes", "Honduras": "Amerique centrale", "Hungary": "Europe",
    "Iceland": "Europe", "India": "Asie", "Indonesia": "Asie", "Iran": "Asie",
    "Iraq": "Asie", "Ireland": "Europe", "Israel": "Asie", "Italy": "Europe",
    "Jamaica": "Caraibes", "Japan": "Asie", "Jordan": "Asie",
    "Kazakhstan": "Asie", "Kenya": "Afrique", "Kyrgyzstan": "Asie", "Laos": "Asie",
    "Latvia": "Europe", "Lebanon": "Asie", "Lesotho": "Afrique",
    "Liberia": "Afrique", "Libya": "Afrique", "Lithuania": "Europe",
    "Luxembourg": "Europe", "Madagascar": "Afrique", "Malawi": "Afrique",
    "Malaysia": "Asie", "Mali": "Afrique", "Mauritania": "Afrique",
    "Mauritius": "Afrique", "Mexico": "Amerique du Nord", "Moldova": "Europe",
    "Mongolia": "Asie", "Montenegro": "Europe", "Morocco": "Afrique",
    "Mozambique": "Afrique", "Myanmar": "Asie", "Namibia": "Afrique",
    "Nepal": "Asie", "Netherlands": "Europe", "New Zealand": "Oceanie",
    "Nicaragua": "Amerique centrale", "Niger": "Afrique", "Nigeria": "Afrique",
    "North Korea": "Asie", "North Macedonia": "Europe", "Norway": "Europe",
    "Oman": "Asie", "Pakistan": "Asie", "Palestine": "Asie",
    "Panama": "Amerique centrale", "Papua New Guinea": "Oceanie",
    "Paraguay": "Amerique du Sud", "Peru": "Amerique du Sud", "Philippines": "Asie",
    "Poland": "Europe", "Portugal": "Europe", "Puerto Rico": "Caraibes",
    "Romania": "Europe", "Russia": "Europe", "Rwanda": "Afrique",
    "Saudi Arabia": "Asie", "Senegal": "Afrique", "Serbia": "Europe",
    "Sierra Leone": "Afrique", "Slovakia": "Europe", "Slovenia": "Europe",
    "Somalia": "Afrique", "South Africa": "Afrique", "South Korea": "Asie",
    "South Sudan": "Afrique", "Spain": "Europe", "Sri Lanka": "Asie",
    "Sudan": "Afrique", "Suriname": "Amerique du Sud", "Sweden": "Europe",
    "Switzerland": "Europe", "Syria": "Asie", "Taiwan": "Asie",
    "Tajikistan": "Asie", "Tanzania": "Afrique", "Thailand": "Asie",
    "Togo": "Afrique", "Trinidad and Tobago": "Caraibes", "Tunisia": "Afrique",
    "Turkey": "Asie", "Turkmenistan": "Asie", "Uganda": "Afrique",
    "Ukraine": "Europe", "United Arab Emirates": "Asie", "United Kingdom": "Europe",
    "United States": "Amerique du Nord", "Uruguay": "Amerique du Sud",
    "Uzbekistan": "Asie", "Venezuela": "Amerique du Sud", "Vietnam": "Asie",
    "Yemen": "Asie", "Zambia": "Afrique", "Zimbabwe": "Afrique",
}
CONTINENT_NAMES = sorted(set(CONTINENT_MAP.values())) + ["Autre"]
CONTINENT_ENC = {name: i for i, name in enumerate(CONTINENT_NAMES)}
TYPE_ENC = {t: i for i, t in enumerate(CLIMATE_TYPES)}


def categoriser_risque(val: float) -> str:
    """Categorise l'impact en niveau de risque ordinal."""
    if val == 0:
        return "Aucun"
    if val < 100:
        return "Faible"
    if val < 1000:
        return "Modere"
    if val < 10000:
        return "Eleve"
    return "Critique"


def charger_dataset() -> pd.DataFrame:
    """Charge le CSV EM-DAT et le pivote en format long (pays x type x annee)."""
    logger.info("Chargement du dataset : %s", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    df_wide = df_raw[~df_raw["entity"].isin(AGGREGATES)].copy()

    value_cols = [f"total_dead_{t}_decadal" for t in CLIMATE_TYPES]
    df = df_wide.melt(
        id_vars=["entity", "code", "year"],
        value_vars=value_cols,
        var_name="disaster_type",
        value_name="impact",
    )
    df["disaster_type"] = (
        df["disaster_type"].str.replace("total_dead_", "").str.replace("_decadal", "")
    )
    df = df.rename(columns={"entity": "country"})
    df["impact"] = df["impact"].fillna(0)

    logger.info(
        "Dataset long : %d lignes, %d pays, %d types",
        len(df), df["country"].nunique(), df["disaster_type"].nunique(),
    )
    return df


def construire_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features (continent, lags par pays x type, warming, cross-type)."""
    df["continent"] = df["country"].map(CONTINENT_MAP).fillna("Autre")
    df["continent_enc"] = df["continent"].map(CONTINENT_ENC)
    df["type_enc"] = df["disaster_type"].map(TYPE_ENC)
    df["warming"] = df["year"].map(WARMING_INDEX)
    df["warming_sq"] = df["warming"] ** 2

    df = df.sort_values(["country", "disaster_type", "year"]).reset_index(drop=True)
    df["log_impact"] = np.log1p(df["impact"])
    df["decade_index"] = (df["year"] - 1900) // 10

    grp = ["country", "disaster_type"]
    df["lag_1"] = df.groupby(grp)["impact"].shift(1)
    df["lag_2"] = df.groupby(grp)["impact"].shift(2)
    df["log_lag_1"] = np.log1p(df["lag_1"])
    df["log_lag_2"] = np.log1p(df["lag_2"])
    df["trend"] = (df["impact"] - df["lag_1"]) / (df["lag_1"] + 1)

    df["cumul_mean"] = df.groupby(grp)["impact"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["cumul_std"] = df.groupby(grp)["impact"].transform(
        lambda x: x.expanding().std().shift(1)
    )
    df["cumul_max"] = df.groupby(grp)["impact"].transform(
        lambda x: x.expanding().max().shift(1)
    )
    df["log_cumul_mean"] = np.log1p(df["cumul_mean"])

    df["impact_other_types"] = (
        df.groupby(["country", "year"])["impact"].transform("sum") - df["impact"]
    )
    df["log_impact_other"] = np.log1p(df["impact_other_types"])

    df["risk_level"] = df["impact"].apply(categoriser_risque)
    return df


REG_MODELS = {
    # Quantile loss (median) = robuste aux outliers (ex: canicule 2003 France)
    "GBM_quantile_median": {
        "class": GradientBoostingRegressor,
        "kwargs": {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
            "loss": "quantile", "alpha": 0.5,
        },
    },
    "GBM_quantile_deep": {
        "class": GradientBoostingRegressor,
        "kwargs": {
            "n_estimators": 400, "max_depth": 7, "learning_rate": 0.05,
            "loss": "quantile", "alpha": 0.5,
        },
    },
    "GBM_quantile_shallow": {
        "class": GradientBoostingRegressor,
        "kwargs": {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
            "loss": "quantile", "alpha": 0.5,
        },
    },
    # Reference : MSE (mean) - sensible aux outliers, garde pour comparaison
    "GBM_mean": {
        "class": GradientBoostingRegressor,
        "kwargs": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    },
    "RF_200": {
        "class": RandomForestRegressor,
        "kwargs": {"n_estimators": 200, "max_depth": 10, "n_jobs": -1},
    },
    "HistGBM": {
        "class": HistGradientBoostingRegressor,
        "kwargs": {"max_iter": 300, "max_depth": 7, "learning_rate": 0.1},
    },
    "Ridge": {"class": Ridge, "kwargs": {"alpha": 1.0}},
    "DecisionTree": {"class": DecisionTreeRegressor, "kwargs": {"max_depth": 10}},
}

CLS_MODELS = {
    "GBM_base": {"class": GradientBoostingClassifier,
                 "kwargs": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}},
    "GBM_deep": {"class": GradientBoostingClassifier,
                 "kwargs": {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.05}},
    "GBM_shallow": {"class": GradientBoostingClassifier,
                    "kwargs": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}},
    "RF_200": {"class": RandomForestClassifier,
               "kwargs": {"n_estimators": 200, "max_depth": 10, "n_jobs": -1,
                          "class_weight": "balanced"}},
    "RF_500": {"class": RandomForestClassifier,
               "kwargs": {"n_estimators": 500, "max_depth": 15, "n_jobs": -1,
                          "class_weight": "balanced"}},
    "HistGBM": {"class": HistGradientBoostingClassifier,
                "kwargs": {"max_iter": 300, "max_depth": 7, "learning_rate": 0.1,
                           "class_weight": "balanced"}},
    "LogReg": {"class": LogisticRegression,
               "kwargs": {"max_iter": 1000, "class_weight": "balanced"}},
    "DecisionTree": {"class": DecisionTreeClassifier,
                     "kwargs": {"max_depth": 10, "class_weight": "balanced"}},
}


class _noop:
    """Context manager inerte quand MLflow est absent."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _build_pipeline(estimator, step_name: str) -> Pipeline:
    """Construit un pipeline SimpleImputer + StandardScaler + estimator."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (step_name, estimator),
    ])


def entrainer_et_exporter(df: pd.DataFrame) -> None:
    """Grid de modeles + export des meilleurs artefacts."""
    df_model = df[df["year"] >= 1920].copy()
    mask_train = df_model["year"] < 2010
    mask_test = df_model["year"] >= 2010

    X_train = df_model.loc[mask_train, FEATURES_REG].values
    X_test = df_model.loc[mask_test, FEATURES_REG].values
    y_train_reg = df_model.loc[mask_train, "log_impact"].values
    y_test_reg = df_model.loc[mask_test, "log_impact"].values
    y_train_cls = df_model.loc[mask_train, "risk_level"].map(RISK_MAP).values
    y_test_cls = df_model.loc[mask_test, "risk_level"].map(RISK_MAP).values
    sw_train = compute_sample_weight("balanced", y_train_cls)

    logger.info("Train : %d lignes | Test : %d lignes | %d features",
                len(X_train), len(X_test), X_train.shape[1])

    # Prepare MLflow context (if available)
    tracking_uri = None
    env_label = "local"
    if HAS_MLFLOW:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(BASE / 'mlflow.db').resolve()}"
        env_label = os.getenv("ENV_LABEL") or ("ci" if os.getenv("CI") else "local")
        mlflow.set_tracking_uri(tracking_uri)
        exp_name = os.getenv("MLFLOW_EXPERIMENT") or "NB10_train_script"
        mlflow.set_experiment(exp_name)

    parent_cm = mlflow.start_run(
        run_name=f"train_nb10_{env_label}_{datetime.now():%Y%m%d_%H%M%S}"
    ) if HAS_MLFLOW else _noop()
    with parent_cm as parent_run:
        if HAS_MLFLOW:
            mlflow.set_tags({
                "git_commit": _git_commit(),
                "git_branch": _git_branch(),
                "env": env_label,
                "python": sys.version.split()[0],
                "sklearn": sklearn.__version__,
                "platform": platform.platform(),
                "trigger": "github_actions" if os.getenv("CI") else "manual",
                "experiment_type": "quantile_pivot",  # bascule median vs mean
                "motivation": "robustesse aux outliers (canicule France 2003)",
            })
            mlflow.log_params({
                "n_features": len(FEATURES_REG),
                "features": str(FEATURES_REG),
                "split": "temporal_2010",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "seed": SEED,
                "n_reg_models": len(REG_MODELS),
                "n_cls_models": len(CLS_MODELS),
                "warming_2030": WARMING_INDEX[2030],
            })

        # ── GRID REGRESSION ────────────────────────────────────────────
        results_reg = []
        # Selection par MAE_test (minimize) : plus robuste que R2 aux outliers
        # Mieux aligne avec Quantile Regression qui cible la mediane
        best_reg, best_reg_name, best_reg_mae = None, None, np.inf
        best_reg_r2 = -np.inf
        for name, spec in REG_MODELS.items():
            t0 = datetime.now()
            kwargs = {**spec["kwargs"]}
            if "random_state" in spec["class"]().get_params():
                kwargs["random_state"] = SEED
            estimator = spec["class"](**kwargs)
            pipe = _build_pipeline(estimator, "reg")
            pipe.fit(X_train, y_train_reg)
            dur = (datetime.now() - t0).total_seconds()

            y_pred_tr = pipe.predict(X_train)
            y_pred_te = pipe.predict(X_test)
            r2_tr = float(r2_score(y_train_reg, y_pred_tr))
            r2_te = float(r2_score(y_test_reg, y_pred_te))
            mae_te = float(mean_absolute_error(y_test_reg, y_pred_te))
            rmse_te = float(np.sqrt(mean_squared_error(y_test_reg, y_pred_te)))

            results_reg.append({"name": name, "r2_test": r2_te, "mae_test": mae_te,
                                "rmse_test": rmse_te, "dur_s": dur})
            logger.info("REG %s : R2_test=%.4f MAE=%.4f (%.1fs)",
                        name, r2_te, mae_te, dur)

            if HAS_MLFLOW:
                # Tag loss family pour tracabilite (mean vs quantile/median vs other)
                loss_kind = spec["kwargs"].get("loss", "mse")
                if loss_kind == "quantile":
                    loss_family = f"quantile_q{spec['kwargs'].get('alpha', 0.5)}"
                elif loss_kind in ("mse", "squared_error"):
                    loss_family = "mean_mse"
                else:
                    loss_family = loss_kind
                with mlflow.start_run(run_name=f"reg_{name}", nested=True):
                    mlflow.set_tags({
                        "loss_family": loss_family,
                        "selection_metric": "mae_test",
                        "pivot_date": "2026-04-15",  # bascule quantile
                    })
                    mlflow.log_params({"model_type": spec["class"].__name__,
                                       "task": "regression", **spec["kwargs"]})
                    mlflow.log_metric("r2_train", r2_tr)
                    mlflow.log_metric("r2_test", r2_te)
                    mlflow.log_metric("mae_test", mae_te)
                    mlflow.log_metric("rmse_test", rmse_te)
                    mlflow.log_metric("train_time_s", dur)

            if mae_te < best_reg_mae:
                best_reg_mae = mae_te
                best_reg_r2 = r2_te
                best_reg = pipe
                best_reg_name = name

        # ── GRID CLASSIFICATION ────────────────────────────────────────
        results_cls = []
        best_cls, best_cls_name, best_cls_f1 = None, None, -np.inf
        for name, spec in CLS_MODELS.items():
            t0 = datetime.now()
            kwargs = {**spec["kwargs"]}
            if "random_state" in spec["class"]().get_params():
                kwargs["random_state"] = SEED
            estimator = spec["class"](**kwargs)
            pipe = _build_pipeline(estimator, "clf")
            try:
                pipe.fit(X_train, y_train_cls, clf__sample_weight=sw_train)
            except (TypeError, ValueError):
                pipe.fit(X_train, y_train_cls)
            dur = (datetime.now() - t0).total_seconds()

            y_pred_te = pipe.predict(X_test)
            acc_te = float(accuracy_score(y_test_cls, y_pred_te))
            f1_te = float(f1_score(y_test_cls, y_pred_te, average="macro"))
            f1_w_te = float(f1_score(y_test_cls, y_pred_te, average="weighted"))

            results_cls.append({"name": name, "accuracy_test": acc_te,
                                "f1_macro_test": f1_te, "dur_s": dur})
            logger.info("CLS %s : acc_test=%.4f f1_macro=%.4f (%.1fs)",
                        name, acc_te, f1_te, dur)

            if HAS_MLFLOW:
                with mlflow.start_run(run_name=f"cls_{name}", nested=True):
                    mlflow.log_params({"model_type": spec["class"].__name__,
                                       "task": "classification", **spec["kwargs"]})
                    mlflow.log_metric("accuracy_test", acc_te)
                    mlflow.log_metric("f1_macro_test", f1_te)
                    mlflow.log_metric("f1_weighted_test", f1_w_te)
                    mlflow.log_metric("train_time_s", dur)

            if f1_te > best_cls_f1:
                best_cls_f1 = f1_te
                best_cls = pipe
                best_cls_name = name

        logger.info(
            "MEILLEUR REG : %s (MAE_test=%.4f, R2_test=%.4f)",
            best_reg_name, best_reg_mae, best_reg_r2,
        )
        logger.info("MEILLEUR CLS : %s (f1_macro=%.4f)", best_cls_name, best_cls_f1)

        # Metriques finales + enregistrement modele registry
        y_pred_reg = best_reg.predict(X_train)
        y_pred_cls = best_cls.predict(X_train)
        metrics = {
            "best_reg_r2_test": best_reg_r2,
            "best_cls_f1_test": best_cls_f1,
            "train_r2": float(r2_score(y_train_reg, y_pred_reg)),
            "train_mae": float(mean_absolute_error(y_train_reg, y_pred_reg)),
            "train_accuracy": float(accuracy_score(y_train_cls, y_pred_cls)),
            "train_f1_macro": float(f1_score(y_train_cls, y_pred_cls, average="macro")),
        }
        pipe_reg = best_reg
        pipe_cls = best_cls

        if HAS_MLFLOW:
            # Metriques globales sur le parent
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.set_tag("best_reg_model", best_reg_name)
            mlflow.set_tag("best_cls_model", best_cls_name)

            # Log & registry des meilleurs modeles
            sig_reg = infer_signature(X_train, y_pred_reg)
            mlflow.sklearn.log_model(
                pipe_reg, name="best_regression",
                signature=sig_reg, input_example=X_train[:3],
                registered_model_name="NB10_regression" if env_label != "ci" else None,
            )
            sig_cls = infer_signature(X_train, y_pred_cls)
            mlflow.sklearn.log_model(
                pipe_cls, name="best_classification",
                signature=sig_cls, input_example=X_train[:3],
                registered_model_name="NB10_classification" if env_label != "ci" else None,
            )

            # Artifacts : plots
            for png in ["NB10_eda.png", "NB10_regression.png",
                        "NB10_classification.png", "NB10_predictions_2030.png"]:
                p = OUTPUT_DIR / png
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="plots")

            # Dataset logging
            try:
                dataset = mlflow.data.from_pandas(
                    df_model.head(1000), source=str(DATA_PATH), name="emdat_decadal",
                )
                mlflow.log_input(dataset, context="training")
            except Exception as e:
                logger.debug("mlflow.log_input skip : %s", e)

            # Tableau comparatif
            pd.DataFrame(results_reg).to_csv(OUTPUT_DIR / "NB10_mlflow_reg_grid.csv", index=False)
            pd.DataFrame(results_cls).to_csv(OUTPUT_DIR / "NB10_mlflow_cls_grid.csv", index=False)
            mlflow.log_artifact(str(OUTPUT_DIR / "NB10_mlflow_reg_grid.csv"))
            mlflow.log_artifact(str(OUTPUT_DIR / "NB10_mlflow_cls_grid.csv"))

            logger.info("MLflow : %d runs reg + %d runs cls + parent sur %s",
                        len(results_reg), len(results_cls), tracking_uri)

    # Predictions 2030 pour chaque (pays, type)
    df_2020 = df[df["year"] == 2020].copy()
    df_2010 = df[df["year"] == 2010].copy()

    pred_rows = []
    for _, row in df_2020.iterrows():
        pays = row["country"]
        typ = row["disaster_type"]
        hist_2010 = df_2010[(df_2010["country"] == pays) & (df_2010["disaster_type"] == typ)]
        lag_2_val = hist_2010.iloc[0]["impact"] if len(hist_2010) > 0 else 0
        all_hist = df[(df["country"] == pays) & (df["disaster_type"] == typ)]
        other_2020 = df_2020[
            (df_2020["country"] == pays) & (df_2020["disaster_type"] != typ)
        ]
        features = {
            "decade_index": 13,
            "continent_enc": row["continent_enc"],
            "type_enc": row["type_enc"],
            "log_lag_1": np.log1p(row["impact"]),
            "log_lag_2": np.log1p(lag_2_val),
            "trend": row["trend"] if pd.notnull(row["trend"]) else 0,
            "log_cumul_mean": np.log1p(all_hist["impact"].mean()),
            "cumul_std": all_hist["impact"].std() if len(all_hist) > 1 else 0,
            "cumul_max": all_hist["impact"].max(),
            "log_impact_other": np.log1p(other_2020["impact"].sum()),
            "warming": WARMING_INDEX[2030],
            "warming_sq": WARMING_INDEX[2030] ** 2,
        }
        pred_rows.append({
            "country": pays, "continent": row["continent"],
            "disaster_type": typ, **features,
        })

    df_pred = pd.DataFrame(pred_rows)
    X_2030 = df_pred[FEATURES_REG].values
    df_pred["log_impact_pred"] = pipe_reg.predict(X_2030)
    df_pred["impact_pred"] = np.expm1(df_pred["log_impact_pred"]).clip(lower=0)
    df_pred["risk_pred_enc"] = pipe_cls.predict(X_2030)
    df_pred["risk_pred"] = df_pred["risk_pred_enc"].map(
        {i: n for i, n in enumerate(RISK_NAMES)}
    )

    df_country = df_pred.groupby(["country", "continent"])["impact_pred"].sum().reset_index()

    # Sauvegarde : 7 joblib + 2 CSV
    joblib.dump(pipe_reg, OUTPUT_DIR / "NB10_best_reg.joblib")
    joblib.dump(pipe_cls, OUTPUT_DIR / "NB10_best_cls.joblib")
    joblib.dump(CONTINENT_ENC, OUTPUT_DIR / "NB10_continent_enc.joblib")
    joblib.dump(TYPE_ENC, OUTPUT_DIR / "NB10_type_enc.joblib")
    joblib.dump(RISK_NAMES, OUTPUT_DIR / "NB10_risk_names.joblib")
    joblib.dump(WARMING_INDEX, OUTPUT_DIR / "NB10_warming_index.joblib")
    joblib.dump(FEATURES_REG, OUTPUT_DIR / "NB10_features.joblib")

    df_pred[["country", "continent", "disaster_type", "impact_pred", "risk_pred"]].to_csv(
        OUTPUT_DIR / "NB10_predictions_2030_detail.csv", index=False
    )
    df_country.sort_values("impact_pred", ascending=False).to_csv(
        OUTPUT_DIR / "NB10_predictions_2030_country.csv", index=False
    )

    logger.info(
        "Artefacts sauvegardes dans %s : 7 joblib + 2 CSV (%d pays x type, %d pays)",
        OUTPUT_DIR, len(df_pred), len(df_country),
    )


def main() -> None:
    """Pipeline complete : chargement, features, entrainement, export."""
    df = charger_dataset()
    df = construire_features(df)
    entrainer_et_exporter(df)
    logger.info("Entrainement NB10 termine avec succes")


if __name__ == "__main__":
    main()
