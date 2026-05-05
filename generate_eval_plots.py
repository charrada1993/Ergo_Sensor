"""
Generate all model evaluation curves for Ergo Sensor AI v2.0
"""
import pandas as pd, numpy as np, json, joblib, warnings, os
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc
)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Load data & models
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('dataset_TMS_enriched.csv')
with open('condition_mappings.json') as f:
    mappings = json.load(f)

FEATURES = [
    'neck','trunk','shoulder','elbow','wrist','hip','knee',
    'r_shoulder','l_shoulder','r_elbow','l_elbow','r_wrist','l_wrist','r_hip','l_hip','r_knee','l_knee',
    'neck_vel','trunk_vel','shoulder_vel','elbow_vel','wrist_vel','hip_vel','knee_vel',
    'neck_duration','trunk_duration','shoulder_duration','elbow_duration','wrist_duration','hip_duration','knee_duration',
    'neck_freq','trunk_freq','shoulder_freq','elbow_freq','wrist_freq','hip_freq','knee_freq',
]

split = int(len(df) * 0.8)
X_tr = df[FEATURES].iloc[:split]
X_te = df[FEATURES].iloc[split:]
y_reg_te  = df['risk_score'].iloc[split:]
y_cls_te  = df['condition_code'].iloc[split:]
y_sev_te  = df['severity_code'].iloc[split:]

scaler    = joblib.load('models/feature_scaler.pkl')
Xs_tr     = pd.DataFrame(scaler.transform(X_tr), columns=FEATURES)
Xs_te     = pd.DataFrame(scaler.transform(X_te), columns=FEATURES)

reg_model = lgb.Booster(model_file='models/lgb_regressor.txt')
cls_model = lgb.Booster(model_file='models/lgb_classifier.txt')
sev_model = lgb.Booster(model_file='models/lgb_severity.txt')
iso       = joblib.load('models/isolation_forest.pkl')
scif      = joblib.load('models/scaler_if.pkl')

pred_reg  = reg_model.predict(Xs_te)
pred_cls  = np.argmax(cls_model.predict(Xs_te), axis=1)
pred_sev  = np.argmax(sev_model.predict(Xs_te), axis=1)

# ─────────────────────────────────────────────────────────────
# Dark theme
# ─────────────────────────────────────────────────────────────
BG     = '#0d0f1a'
CARD   = '#121428'
ACCENT = '#00e5ff'
WARN   = '#ffaa00'
DANGER = '#ff4d6d'
OK     = '#00e5a0'
PURPLE = '#c97fff'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   CARD,
    'text.color':       'white',
    'axes.labelcolor':  'white',
    'xtick.color':      '#8890aa',
    'ytick.color':      '#8890aa',
    'axes.edgecolor':   '#2a2d4a',
    'grid.color':       '#1e2038',
    'font.family':      'monospace',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

fig = plt.figure(figsize=(22, 28), facecolor=BG)
fig.suptitle('ERGO SENSOR  |  AI MODELS — FULL EVALUATION DASHBOARD  (v2.0)',
             fontsize=17, fontweight='bold', color=ACCENT, y=0.995)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.36)

# ─────────────────────────────────────────────────────────────
# 1. Actual vs Predicted — Regression
# ─────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
idx = np.argsort(y_reg_te.values)
ax1.plot(y_reg_te.values[idx], color='#4a5275', lw=1.5, label='Actual', alpha=0.9)
ax1.plot(pred_reg[idx],        color=ACCENT,    lw=1.0, label='Predicted', alpha=0.8)
r2  = r2_score(y_reg_te, pred_reg)
mae = mean_absolute_error(y_reg_te, pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_te, pred_reg))
ax1.set_title('LightGBM Regression — Actual vs Predicted Risk Score', color=ACCENT, fontsize=11, pad=10)
ax1.set_xlabel('Sample Index (sorted by actual)'); ax1.set_ylabel('Risk Score (0–1)')
ax1.legend(facecolor=BG, edgecolor='#2a2d4a')
ax1.grid(True, alpha=0.18)
ax1.text(0.98, 0.06,
         f'R² = {r2:.4f}    MAE = {mae:.4f}    RMSE = {rmse:.4f}',
         transform=ax1.transAxes, ha='right', color=OK, fontsize=10,
         bbox=dict(facecolor='#0d0f1a', edgecolor='#2a2d4a', boxstyle='round,pad=0.4'))

# ─────────────────────────────────────────────────────────────
# 2. Residuals Histogram
# ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
residuals = pred_reg - y_reg_te.values
ax2.hist(residuals, bins=70, color=ACCENT, alpha=0.75, edgecolor='none')
ax2.axvline(0, color=DANGER, lw=2, ls='--', label='Zero error')
ax2.axvline(residuals.mean(), color=WARN, lw=1.5, ls=':', label=f'Mean={residuals.mean():.4f}')
ax2.set_title('Residuals Distribution', color=ACCENT, fontsize=11, pad=10)
ax2.set_xlabel('Prediction Error'); ax2.set_ylabel('Count')
ax2.legend(facecolor=BG, edgecolor='#2a2d4a', fontsize=8)
ax2.grid(True, alpha=0.18)

# ─────────────────────────────────────────────────────────────
# 3. Condition Classifier — Confusion Matrix
# ─────────────────────────────────────────────────────────────
cond_labels = {v: k for k, v in mappings['condition_to_code'].items()}
n_cls = len(mappings['condition_to_code'])
cm_cls = confusion_matrix(y_cls_te, pred_cls)

ax3 = fig.add_subplot(gs[1, :2])
im3 = ax3.imshow(cm_cls, cmap='YlOrRd', aspect='auto', interpolation='nearest')
short = [cond_labels.get(i, '?')[:12] for i in range(n_cls)]
ax3.set_xticks(range(n_cls)); ax3.set_xticklabels(short, rotation=45, ha='right', fontsize=6.5)
ax3.set_yticks(range(n_cls)); ax3.set_yticklabels(short, fontsize=6.5)
plt.colorbar(im3, ax=ax3, fraction=0.025, pad=0.02)
acc_cls = accuracy_score(y_cls_te, pred_cls)
f1_cls  = f1_score(y_cls_te, pred_cls, average='macro', zero_division=0)
ax3.set_title(f'Condition Classifier — Confusion Matrix  [Acc={acc_cls:.4f}  F1={f1_cls:.4f}]',
              color=ACCENT, fontsize=10, pad=10)
ax3.set_xlabel('Predicted Class', color='white')
ax3.set_ylabel('True Class', color='white')

# ─────────────────────────────────────────────────────────────
# 4. Severity Confusion Matrix
# ─────────────────────────────────────────────────────────────
sev_labels = {v: k for k, v in mappings['severity_to_code'].items()}
cm_sev = confusion_matrix(y_sev_te, pred_sev)
ax4 = fig.add_subplot(gs[1, 2])
sns.heatmap(cm_sev, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['low', 'medium', 'high'],
            yticklabels=['low', 'medium', 'high'],
            cbar=False, linewidths=0.8, linecolor='#0d0f1a',
            annot_kws={'color': 'white', 'fontsize': 12})
acc_sev = accuracy_score(y_sev_te, pred_sev)
f1_sev  = f1_score(y_sev_te, pred_sev, average='macro', zero_division=0)
ax4.set_title(f'Severity Model  [Acc={acc_sev:.4f}  F1={f1_sev:.4f}]', color=ACCENT, fontsize=10, pad=10)
ax4.set_xlabel('Predicted', color='white'); ax4.set_ylabel('Actual', color='white')
ax4.tick_params(colors='#8890aa')

# ─────────────────────────────────────────────────────────────
# 5. Feature Importance — Top 20
# ─────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
fi     = reg_model.feature_importance(importance_type='gain')
fi_df  = pd.DataFrame({'feature': FEATURES, 'gain': fi}).sort_values('gain', ascending=True).tail(20)
max_g  = fi_df['gain'].max()
colors = [DANGER if v > max_g * 0.65 else WARN if v > max_g * 0.30 else ACCENT for v in fi_df['gain']]
ax5.barh(fi_df['feature'], fi_df['gain'], color=colors, alpha=0.88, edgecolor='none', height=0.7)
ax5.set_title('LightGBM Feature Importance — Top 20 (Information Gain)', color=ACCENT, fontsize=11, pad=10)
ax5.set_xlabel('Gain Score')
ax5.grid(True, axis='x', alpha=0.18)
for i, (f_, g_) in enumerate(zip(fi_df['feature'], fi_df['gain'])):
    ax5.text(g_ + max_g * 0.005, i, f'{g_:,.0f}', va='center', color='#8890aa', fontsize=7)

# ─────────────────────────────────────────────────────────────
# 6. IsolationForest Score Distribution
# ─────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
iso_scores  = iso.decision_function(scif.transform(Xs_te))
iso_pred    = iso.predict(scif.transform(Xs_te))
normal_mask = iso_pred == 1
ax6.hist(iso_scores[normal_mask],  bins=50, color=OK,     alpha=0.75, label='Normal',  edgecolor='none')
ax6.hist(iso_scores[~normal_mask], bins=25, color=DANGER, alpha=0.85, label='Anomaly', edgecolor='none')
ax6.axvline(0, color='white', lw=1.2, ls='--', alpha=0.5)
ax6.set_title('IsolationForest — Decision Score', color=ACCENT, fontsize=11, pad=10)
ax6.set_xlabel('Score (negative = anomaly)'); ax6.set_ylabel('Count')
ax6.legend(facecolor=BG, edgecolor='#2a2d4a')
ax6.grid(True, alpha=0.18)
n_anom = (~normal_mask).sum()
ax6.text(0.98, 0.92, f'Anomalies: {n_anom} ({100*n_anom/len(iso_pred):.1f}%)',
         transform=ax6.transAxes, ha='right', color=DANGER, fontsize=9)

# ─────────────────────────────────────────────────────────────
# 7. Anomaly Models — ROC Curves
# ─────────────────────────────────────────────────────────────
ANOMALY_COLS = [
    'anomaly_neck_hyperflex',
    'anomaly_shoulder_overext',
    'anomaly_wrist_strain',
    'anomaly_trunk_torsion',
    'anomaly_elbow_hyperext',
]
ANOMALY_SRCS   = ['neck', 'shoulder', 'wrist', 'trunk', 'elbow']
ANOM_COLORS    = [ACCENT, OK, WARN, DANGER, PURPLE]
ANOM_LABELS    = ['Neck Hyperflexion', 'Shoulder Overext', 'Wrist Strain', 'Trunk Torsion', 'Elbow Hyperext']

ax7 = fig.add_subplot(gs[3, :])
ax7.plot([0, 1], [0, 1], color='#4a5275', ls='--', lw=1.2, label='Random (AUC=0.5)')

for col, src, clr, lbl in zip(ANOMALY_COLS, ANOMALY_SRCS, ANOM_COLORS, ANOM_LABELS):
    try:
        am    = lgb.Booster(model_file=f'models/lgbm_{col}.txt')
        thr   = df[src].iloc[:split].quantile(0.95)
        y_bin = (df[src].iloc[split:] > thr).astype(int).values
        y_sc  = am.predict(Xs_te)
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        roc_auc     = auc(fpr, tpr)
        ax7.plot(fpr, tpr, color=clr, lw=2.2,
                 label=f'{lbl}  (AUC = {roc_auc:.3f})')
    except Exception as e:
        print(f'[WARN] {col}: {e}')

ax7.set_title('Anomaly Models — ROC Curves (per articulation)', color=ACCENT, fontsize=12, pad=12)
ax7.set_xlabel('False Positive Rate'); ax7.set_ylabel('True Positive Rate')
ax7.legend(facecolor=BG, edgecolor='#2a2d4a', loc='lower right', fontsize=10)
ax7.grid(True, alpha=0.18)
ax7.fill_between([0, 1], [0, 1], alpha=0.04, color='white')

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
out = 'plots/model_evaluation.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'[OK] Saved: {out}')

# ─────────────────────────────────────────────────────────────
# Metrics summary JSON (for ai.html)
# ─────────────────────────────────────────────────────────────
summary = {
    "regression":   {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)},
    "condition":    {"accuracy": round(acc_cls, 4), "f1_macro": round(f1_cls, 4)},
    "severity":     {"accuracy": round(acc_sev, 4), "f1_macro": round(f1_sev, 4)},
    "isolation_forest": {"anomaly_rate": round(n_anom / len(iso_pred), 4)},
    "n_features":   len(FEATURES),
    "n_test":       len(Xs_te),
}
with open('plots/metrics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('[OK] Saved: plots/metrics_summary.json')
print(json.dumps(summary, indent=2))
