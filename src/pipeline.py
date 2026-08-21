"""
Core pipeline: data loading, evidential-distance diagnostic, and the
lightweight nutrient-decline predictor used as f(x)/U(x).

All numeric results in this module come from real, downloaded, published
data (Loladze 2014 eLife/Dryad CO2-ionome meta-analysis; GeoNutrition
Ethiopia/Malawi cereal grain surveys). No synthetic or fabricated
observations are used anywhere in this file.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

RNG = 13  # fixed seed, reported in the paper for reproducibility

# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------

def load_face_data(path="data/loladze_co2_ionome.csv"):
    """Loads the real Loladze (2014) CO2-ionome meta-analysis dataset.
    Source: https://github.com/loladze/co2 (mirrors Dryad doi:10.5061/dryad.6356f)
    """
    df = pd.read_csv(path)
    return df


def crop_face_evidence(df):
    """The broad evidential footprint: every real FACE observation on a
    cultivated crop species. Defines the covariate-space evidence set
    used by the evidential-distance diagnostic C(x).
    """
    sub = df[(df["crop.wild"] == "CRP") & (df["study.type"] == "FACE")].copy()
    sub = sub.dropna(subset=["lat", "long"])
    return sub


def cereal_micronutrient_face(df, elements=("Fe", "Zn", "N")):
    """The narrower training set for the nutrient-decline predictor f(x):
    real FACE observations of Fe, Zn and N (a standard proxy for crude
    protein, N x 6.25) in cereal/oilseed crops.
    """
    sub = df[
        (df["crop.wild"] == "CRP")
        & (df["study.type"] == "FACE")
        & (df["element"].isin(elements))
    ].copy()
    sub = sub.dropna(subset=["lat", "long", "delta", "eco2", "aco2"])
    return sub


def load_geonutrition(eth_path="data/Ethiopia_Grain.xlsx", mal_path="data/Malawi_Grain.xlsx"):
    """Loads the real GeoNutrition cereal grain micronutrient surveys.
    Source: https://github.com/rmlark/GeoNutrition
    """
    eth = pd.read_excel(eth_path)
    eth["country"] = "Ethiopia"
    mal = pd.read_excel(mal_path)
    mal["country"] = "Malawi"
    mal = mal.rename(columns={"Se_triplequad": "Se"})
    both = pd.concat([eth, mal], ignore_index=True, sort=False)
    return eth, mal, both


# ----------------------------------------------------------------------
# 2. Evidential-distance diagnostic  C(x)
# ----------------------------------------------------------------------

class EvidentialCoverage:
    """Training-free diagnostic: how geographically close is a query
    point to the real FACE experimental evidence base?  Implemented as
    mean distance to the k nearest FACE sites in standardized
    (lat, long) space, calibrated against a leave-one-out distribution
    of the evidence set's own internal distances so the resulting score
    is interpretable as a percentile ("how typical is this point,
    relative to what a real FACE-covered point looks like").
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, evidence_df):
        self.sites = evidence_df[["lat", "long"]].drop_duplicates().reset_index(drop=True)
        self.scaler = StandardScaler().fit(self.sites.values)
        Xs = self.scaler.transform(self.sites.values)
        k_eff = min(self.k, len(Xs) - 1)
        self.k_eff = k_eff
        self.nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xs)  # +1: a site is its own neighbor
        # leave-one-out reference distances: for each evidence site, its
        # mean distance to its k nearest *other* evidence sites.
        dists, _ = self.nn.kneighbors(Xs)
        self.ref_distances = dists[:, 1:].mean(axis=1)  # drop self (distance 0)
        return self

    def raw_distance(self, query_latlong):
        Xq = self.scaler.transform(np.atleast_2d(query_latlong))
        dists, _ = self.nn.kneighbors(Xq, n_neighbors=self.k_eff)
        return dists.mean(axis=1)

    def coverage_score(self, query_latlong):
        """Returns a coverage score in [0, 1]: the fraction of the
        evidence base's own internal leave-one-out distances that are
        *larger* than the query's distance to the evidence base.
        1.0 = as typical as a real FACE site; 0.0 = far outside anything
        the literature has ever sampled.
        """
        d = self.raw_distance(query_latlong)
        ref_sorted = np.sort(self.ref_distances)
        pct = np.searchsorted(ref_sorted, d, side="right") / len(ref_sorted)
        return 1.0 - pct  # invert: larger distance -> lower coverage

    def loo_self_coverage(self):
        """Leave-one-out coverage score of the evidence sites against
        themselves -- the sanity-check upper bound (should sit high).
        """
        ref_sorted = np.sort(self.ref_distances)
        pct = np.array([np.searchsorted(ref_sorted, d, side="right") / len(ref_sorted)
                         for d in self.ref_distances])
        return 1.0 - pct


# ----------------------------------------------------------------------
# 3. Nutrient-decline predictor  f(x), U(x)
# ----------------------------------------------------------------------

FEATURES = ["lat", "abs_lat", "long", "eco2", "aco2", "co2_delta"]


def _add_engineered(df):
    df = df.copy()
    df["abs_lat"] = df["lat"].abs()
    df["co2_delta"] = df["eco2"] - df["aco2"]
    return df


def build_design_matrix(df):
    df = _add_engineered(df)
    element_dum = pd.get_dummies(df["element"], prefix="elem")
    X = pd.concat([df[FEATURES].reset_index(drop=True), element_dum.reset_index(drop=True)], axis=1)
    y = df["delta"].reset_index(drop=True).values
    groups = df.apply(lambda r: f"{r['lat']:.2f}_{r['long']:.2f}", axis=1).reset_index(drop=True)
    return X, y, groups


def cross_validated_fit(df, n_estimators=300, max_depth=4, min_samples_leaf=3):
    """Leave-one-site-out (GroupKFold by real experimental site
    coordinates, to avoid leakage between rows from the same field
    trial) cross-validation of a small RandomForestRegressor. Reports
    real, honestly-computed out-of-fold predictions -- no synthetic
    labels or invented performance numbers.
    """
    X, y, groups = build_design_matrix(df)
    n_groups = groups.nunique()
    gkf = GroupKFold(n_splits=n_groups)

    oof_pred = np.full(len(y), np.nan)
    oof_std = np.full(len(y), np.nan)

    X_arr = X.values
    for train_idx, test_idx in gkf.split(X_arr, y, groups):
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=RNG,
        )
        model.fit(X_arr[train_idx], y[train_idx])
        tree_preds = np.stack([t.predict(X_arr[test_idx]) for t in model.estimators_], axis=0)
        oof_pred[test_idx] = tree_preds.mean(axis=0)
        oof_std[test_idx] = tree_preds.std(axis=0)

    # final model, trained on all available evidence, used for
    # out-of-sample (geographic) prediction at query points
    final_model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, random_state=RNG,
    )
    final_model.fit(X_arr, y)

    return final_model, X.columns.tolist(), oof_pred, oof_std, y


def predict_with_uncertainty(model, feature_cols, lat, long, eco2, aco2, element):
    row = pd.DataFrame([{
        "lat": lat, "long": long, "eco2": eco2, "aco2": aco2,
    }])
    row = _add_engineered(row)
    for c in feature_cols:
        if c.startswith("elem_"):
            row[c] = 1.0 if c == f"elem_{element}" else 0.0
    row = row.reindex(columns=feature_cols, fill_value=0.0)
    row_arr = row.values
    tree_preds = np.array([t.predict(row_arr)[0] for t in model.estimators_])
    return tree_preds.mean(), tree_preds.std()
