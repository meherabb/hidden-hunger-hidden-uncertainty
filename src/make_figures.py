"""
Generates all figures for "Hidden Hunger, Hidden Uncertainty" from real,
downloaded data. Every number plotted here is computed in this script --
nothing is hardcoded except geographic reference coordinates (which are
just longitude/latitude facts, not experimental data) used to illustrate
where the diagnostic would be applied.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

import pipeline as pl
from style import set_style, PALETTE, panel_label

set_style()
FIGDIR = "figures"
RESULTS = {}

# ----------------------------------------------------------------------
# Load real data
# ----------------------------------------------------------------------
df = pl.load_face_data()
crop_ev = pl.crop_face_evidence(df)          # 295 rows, 14 sites, 6 countries
cereal = pl.cereal_micronutrient_face(df)     # 76 rows, 11 sites, Fe/Zn/N
eth, mal, both = pl.load_geonutrition()

wheat_eth = eth[eth["Crop"] == "Wheat"][["Latitude", "Longitude"]].dropna()
rice_mal = mal[mal["Crop"] == "Rice"][["Latitude", "Longitude"]].dropna()

cov = pl.EvidentialCoverage(k=5).fit(crop_ev)

RESULTS["n_face_crop_obs"] = int(len(crop_ev))
RESULTS["n_face_crop_sites"] = int(crop_ev[["lat", "long"]].drop_duplicates().shape[0])
RESULTS["face_countries"] = sorted(crop_ev["country"].unique().tolist())
RESULTS["n_cereal_micronutrient_obs"] = int(len(cereal))
RESULTS["n_cereal_micronutrient_sites"] = int(cereal[["lat", "long"]].drop_duplicates().shape[0])
RESULTS["n_geonutrition_eth_wheat"] = int(len(wheat_eth))
RESULTS["n_geonutrition_mal_rice"] = int(len(rice_mal))

# ----------------------------------------------------------------------
# FIGURE 1 -- the evidential gap (2 panels, single row)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

ax = axes[0]
for country, sub in crop_ev.groupby("country"):
    pts = sub[["lat", "long"]].drop_duplicates()
    ax.scatter(pts["long"], pts["lat"], s=22, color=PALETTE["face"], alpha=0.85,
               edgecolor="white", linewidth=0.4, zorder=3)
ax.scatter(wheat_eth["Longitude"], wheat_eth["Latitude"], s=6, color=PALETTE["geonutrition"],
           alpha=0.55, label="GeoNutrition: Ethiopia (wheat)", zorder=2)
ax.scatter(rice_mal["Longitude"], rice_mal["Latitude"], s=6, color=PALETTE["query"],
           alpha=0.55, label="GeoNutrition: Malawi (rice)", zorder=2)
ax.scatter([], [], s=22, color=PALETTE["face"], edgecolor="white", label="FACE evidence site (crop)")
ax.axhline(0, color="#999999", linewidth=0.5, zorder=1)
ax.set_xlim(-125, 155)
ax.set_ylim(-45, 65)
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_title("Where the evidence is vs.\nwhere the dependence is")
ax.legend(loc="lower left", fontsize=6.2, markerscale=1.0)
panel_label(ax, "a")

ax = axes[1]
counts = cereal["country"].value_counts().sort_values(ascending=True)
bars = ax.barh(counts.index, counts.values, color=PALETTE["face"], height=0.6)
ax.set_xlabel("Fe / Zn / N observations\n(real FACE trials, cereals)")
ax.set_title(f"All {RESULTS['n_cereal_micronutrient_obs']} cereal micronutrient\nFACE observations, {len(RESULTS['face_countries'])} countries total")
for b, v in zip(bars, counts.values):
    ax.text(v + 0.4, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=7)
ax.set_xlim(0, counts.values.max() * 1.25)
panel_label(ax, "b")

fig.savefig(f"{FIGDIR}/fig1_evidential_gap.pdf")
plt.close(fig)
print("Figure 1 saved.")

# ----------------------------------------------------------------------
# FIGURE 3 -- coverage separation, model CV fit, confidence-evidence mismatch
# (kept as "Figure 3" to match the paper plan; Figure 2 is the schematic,
#  produced separately since it is not data-driven)
# ----------------------------------------------------------------------
loo_self = cov.loo_self_coverage()
d_eth = cov.raw_distance(wheat_eth.values)
d_mal = cov.raw_distance(rice_mal.values)
d_face_internal = cov.ref_distances  # raw LOO distances among the 14 evidence sites themselves

RESULTS["median_raw_distance_face_internal"] = float(np.median(d_face_internal))
RESULTS["median_raw_distance_ethiopia_wheat"] = float(np.median(d_eth))
RESULTS["median_raw_distance_malawi_rice"] = float(np.median(d_mal))

model, feat_cols, oof_pred, oof_std, y_true = pl.cross_validated_fit(cereal)
mask = ~np.isnan(oof_pred)
cv_r2 = r2_score(y_true[mask], oof_pred[mask])
cv_mae = mean_absolute_error(y_true[mask], oof_pred[mask])
RESULTS["cv_r2_leave_one_site_out"] = float(cv_r2)
RESULTS["cv_mae_leave_one_site_out"] = float(cv_mae)
RESULTS["cv_n_sites"] = int(cereal[["lat", "long"]].drop_duplicates().shape[0])

# Query points: real FACE-internal sites (a sample) + GeoNutrition regions +
# a few additional real-world cereal-growing/consuming regions (coordinates
# are geographic reference points, not experimental data).
extra_regions = {
    "India (Punjab, wheat belt)": (31.1, 75.3),
    "Bangladesh (Dhaka, rice)": (23.8, 90.4),
    "Vietnam (Mekong Delta, rice)": (10.0, 105.8),
    "Nigeria (Kano, cereals)": (12.0, 8.5),
}
query_points = []
for _, row in crop_ev[["lat", "long", "country"]].drop_duplicates().iterrows():
    query_points.append((f"FACE site ({row['country']})", row["lat"], row["long"], "face"))
query_points.append(("GeoNutrition: Ethiopia (wheat, mean loc.)", wheat_eth["Latitude"].mean(),
                      wheat_eth["Longitude"].mean(), "geonutrition"))
query_points.append(("GeoNutrition: Malawi (rice, mean loc.)", rice_mal["Latitude"].mean(),
                      rice_mal["Longitude"].mean(), "geonutrition"))
for name, (lat, lon) in extra_regions.items():
    query_points.append((name, lat, lon, "query"))

rows = []
for name, lat, lon, kind in query_points:
    coverage = cov.coverage_score(np.array([[lat, lon]]))[0]
    raw_d = cov.raw_distance(np.array([[lat, lon]]))[0]
    pred_mean, pred_std = pl.predict_with_uncertainty(
        model, feat_cols, lat=lat, long=lon, eco2=550, aco2=410, element="Fe"
    )
    rows.append(dict(name=name, lat=lat, lon=lon, kind=kind, coverage=coverage,
                      raw_distance=raw_d, pred_decline=pred_mean, pred_std=pred_std))
query_df = pd.DataFrame(rows)
query_df.to_csv("results/query_predictions.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))

ax = axes[0]
groups_to_plot = [
    ("FACE evidence\n(leave-one-out)", d_face_internal, PALETTE["face"]),
    ("Ethiopia\n(wheat)", d_eth, PALETTE["geonutrition"]),
    ("Malawi\n(rice)", d_mal, PALETTE["query"]),
]
positions = [1, 2, 3]
parts = ax.violinplot([g[1] for g in groups_to_plot], positions=positions, showmedians=True, widths=0.75)
for pc, (_, _, color) in zip(parts["bodies"], groups_to_plot):
    pc.set_facecolor(color)
    pc.set_alpha(0.55)
for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
    parts[key].set_color("#444444")
    parts[key].set_linewidth(0.8)
ax.set_xticks(positions)
ax.set_xticklabels([g[0] for g in groups_to_plot], fontsize=7)
ax.set_ylabel("Evidential distance\n(standardized lat/long units)")
ax.set_title("Distance to the real FACE\nevidence base")
panel_label(ax, "a")

ax = axes[1]
ax.scatter(y_true[mask], oof_pred[mask], s=22, color=PALETTE["face"], alpha=0.75, edgecolor="white", linewidth=0.3)
lims = [min(y_true[mask].min(), oof_pred[mask].min()) - 0.05,
        max(y_true[mask].max(), oof_pred[mask].max()) + 0.05]
ax.plot(lims, lims, color="#999999", linewidth=0.8, linestyle="--")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed fractional change")
ax.set_ylabel("Predicted (leave-one-\nsite-out CV)")
ax.set_title(f"Predictor fit on its own\nevidence: R²={cv_r2:.2f}, MAE={cv_mae:.2f}")
panel_label(ax, "b")

ax = axes[2]
color_map = {"face": PALETTE["face"], "geonutrition": PALETTE["geonutrition"], "query": PALETTE["query"]}
for kind, sub in query_df.groupby("kind"):
    ax.errorbar(sub["raw_distance"], sub["pred_decline"], yerr=sub["pred_std"], fmt="o",
                ms=4.5, color=color_map[kind], alpha=0.8, capsize=2, linewidth=0.8,
                label={"face": "FACE-internal site", "geonutrition": "GeoNutrition region",
                       "query": "Other cereal-growing region"}[kind])
dist_std_corr = float(query_df["raw_distance"].corr(query_df["pred_std"]))
coverage_std_corr = float(query_df["coverage"].corr(query_df["pred_std"]))
RESULTS["coverage_vs_pred_std_correlation"] = coverage_std_corr
RESULTS["distance_vs_pred_std_correlation"] = dist_std_corr
ax.set_xlabel("Evidential distance\n(0 = at a FACE site, higher = novel)")
ax.set_ylabel("Predicted Fe decline\n(fractional, ±1 tree-ensemble s.d.)")
ax.set_title(f"Ensemble uncertainty tracks distance\nweakly (r={dist_std_corr:.2f})", fontsize=9)
ax.legend(loc="lower right", fontsize=6.0)
panel_label(ax, "c")

fig.savefig(f"{FIGDIR}/fig3_mismatch.pdf")
plt.close(fig)
print("Figure 3 saved.")

# ----------------------------------------------------------------------
# FIGURE 4 -- robustness to the diagnostic's hyperparameter k
# ----------------------------------------------------------------------
ks = [2, 3, 5, 8, 10]
gap_medians = []
for k in ks:
    cov_k = pl.EvidentialCoverage(k=k).fit(crop_ev)
    d_face_k = cov_k.ref_distances
    d_eth_k = cov_k.raw_distance(wheat_eth.values)
    d_mal_k = cov_k.raw_distance(rice_mal.values)
    gap_medians.append({
        "k": k,
        "face_internal": float(np.median(d_face_k)),
        "ethiopia": float(np.median(d_eth_k)),
        "malawi": float(np.median(d_mal_k)),
    })
gap_df = pd.DataFrame(gap_medians)
gap_df.to_csv("results/robustness_by_k.csv", index=False)
RESULTS["robustness_k_values"] = ks

fig, ax = plt.subplots(1, 1, figsize=(4.4, 2.9))
ax.plot(gap_df["k"], gap_df["face_internal"], marker="o", ms=4, color=PALETTE["face"],
        label="FACE evidence (internal)")
ax.plot(gap_df["k"], gap_df["ethiopia"], marker="o", ms=4, color=PALETTE["geonutrition"],
        label="Ethiopia (wheat)")
ax.plot(gap_df["k"], gap_df["malawi"], marker="o", ms=4, color=PALETTE["query"],
        label="Malawi (rice)")
ax.set_xlabel("k (nearest evidence sites averaged)")
ax.set_ylabel("Median evidential distance")
ax.set_title("The coverage gap is stable\nacross the diagnostic's k")
ax.legend(fontsize=7)
fig.savefig(f"{FIGDIR}/fig4_robustness.pdf")
plt.close(fig)
print("Figure 4 saved.")

# ----------------------------------------------------------------------
# FIGURE 5 -- extrapolation targets: predicted decline vs. coverage
#             for real-world cereal-growing/consuming regions
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4))
order = query_df.sort_values("raw_distance", ascending=True)
colors = order["kind"].map(color_map)
ax.barh(order["name"], order["raw_distance"], color=colors, height=0.6)
ax.set_xlabel("Evidential distance (this work's diagnostic;\nhigher = farther from any real FACE trial)")
ax.set_title("Evidential distance by region\n(FAO dependence weighting pending, see text)", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig5_coverage_by_region.pdf")
plt.close(fig)
print("Figure 5 saved.")

# ----------------------------------------------------------------------
# FIGURE 2 -- method schematic (illustrative; not data-driven)
# ----------------------------------------------------------------------
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(7.4, 2.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 2.9)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.32", linewidth=0.9)
# (center_x, center_y, half_width, half_height, text, color)
boxes = [
    (1.0, 2.2, 0.9, 0.42, "Real FACE evidence\n(lat/long, CO$_2$ level)", PALETTE["face"]),
    (1.0, 0.6, 0.9, 0.42, "Target region\n(lat/long, CO$_2$ level)", PALETTE["query"]),
    (3.55, 2.2, 0.9, 0.42, "Evidential coverage\nC(x)  [training-free]", PALETTE["face"]),
    (3.55, 0.6, 0.9, 0.42, "Decline predictor\nf(x), U(x)  [CV RF]", PALETTE["query"]),
    (6.35, 1.4, 1.05, 0.48, "Confidence-evidence\nrelationship", PALETTE["geonutrition"]),
    (9.05, 1.4, 0.85, 0.42, "Regional\nrisk view", PALETTE["highlight"]),
]
for x, y, hw, hh, text, color in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh,
                                          facecolor=color, alpha=0.18, edgecolor=color, **box_style))
    ax.text(x, y, text, ha="center", va="center", fontsize=7.3)

arrows = [
    (1.0 + 0.9, 2.2, 3.55 - 0.9, 2.2),
    (1.0 + 0.9, 0.6, 3.55 - 0.9, 0.6),
    (3.55 + 0.9, 2.2, 6.35 - 1.05, 1.6),
    (3.55 + 0.9, 0.6, 6.35 - 1.05, 1.2),
    (6.35 + 1.05, 1.4, 9.05 - 0.85, 1.4),
]
for x1, y1, x2, y2 in arrows:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                  color="#555555", linewidth=0.8, shrinkA=2, shrinkB=2))

fig.savefig(f"{FIGDIR}/fig2_method_schematic.pdf")
plt.close(fig)
print("Figure 2 saved.")

with open("results/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(json.dumps(RESULTS, indent=2))
