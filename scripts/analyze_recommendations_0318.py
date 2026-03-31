import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon

plt.style.use('ggplot')

BASE = Path('logs/query_0317')
INPUT = BASE / 'recommendations_extracted.csv'
OUT = BASE / 'analysis_0318'
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- Load & Feature Engineering --------------------
df = pd.read_csv(INPUT)

def parse_need(code: str) -> str:
    c = str(code)
    if c.startswith('H'):
        return 'Hedonic'
    if c.startswith('U'):
        return 'Utilitarian'
    if c.startswith('B'):
        return 'Balanced'
    return 'Unknown'

def parse_expr(code: str) -> str:
    c = str(code)
    if c.endswith('S'):
        return 'Specific'
    if c.endswith('A'):
        return 'Ambiguous'
    return 'Unknown'

df['Need_Type'] = df['code'].map(parse_need)
df['Expression_Type'] = df['code'].map(parse_expr)
df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
df = df.dropna(subset=['steps', 'mode', 'product_id'])

# -------------------- Stage 1: Descriptive & Efficiency --------------------
stage1_mode = df.groupby('mode', as_index=False)['steps'].mean().rename(columns={'steps':'avg_steps'})
stage1_need_mode = df.groupby(['Need_Type','mode'], as_index=False)['steps'].mean().rename(columns={'steps':'avg_steps'})
stage1_expr_mode = df.groupby(['Expression_Type','mode'], as_index=False)['steps'].mean().rename(columns={'steps':'avg_steps'})

stage1_mode.to_csv(OUT / 'stage1_avg_steps_by_mode.csv', index=False)
stage1_need_mode.to_csv(OUT / 'stage1_avg_steps_by_need_mode.csv', index=False)
stage1_expr_mode.to_csv(OUT / 'stage1_avg_steps_by_expr_mode.csv', index=False)

plt.figure(figsize=(6,4))
plt.bar(stage1_mode['mode'], stage1_mode['avg_steps'])
plt.title('Average Steps by Mode')
plt.ylabel('avg_steps')
plt.tight_layout()
plt.savefig(OUT / 'stage1_bar_avg_steps_mode.png', dpi=180)
plt.close()

plt.figure(figsize=(8,4))
pivot_need = stage1_need_mode.pivot(index='Need_Type', columns='mode', values='avg_steps').fillna(0)
x = np.arange(len(pivot_need.index))
width = 0.35
modes = list(pivot_need.columns)
for i, m in enumerate(modes):
    plt.bar(x + (i - (len(modes)-1)/2)*width, pivot_need[m].values, width=width, label=m)
plt.xticks(x, pivot_need.index)
plt.title('Average Steps by Need Type and Mode')
plt.ylabel('avg_steps')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'stage1_bar_avg_steps_need_mode.png', dpi=180)
plt.close()

plt.figure(figsize=(8,4))
need_levels = ['Hedonic', 'Utilitarian', 'Balanced']
modes = sorted(df['mode'].unique())
positions = []
data = []
labels = []
for i, need in enumerate(need_levels):
    for j, m in enumerate(modes):
        vals = df[(df['Need_Type'] == need) & (df['mode'] == m)]['steps'].values
        if len(vals) == 0:
            continue
        positions.append(i*3 + j + 1)
        data.append(vals)
        labels.append(f'{need[:1]}-{m[:1].upper()}')
plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
plt.xticks(positions, labels, rotation=45)
plt.title('Steps Distribution by Need Type and Mode')
plt.ylabel('steps')
plt.tight_layout()
plt.savefig(OUT / 'stage1_box_steps_need_mode.png', dpi=180)
plt.close()

# -------------------- Stage 2: Categorical Association --------------------
# product_id x Need_Type
ct = pd.crosstab(df['product_id'], df['Need_Type'])
chi2, p, dof, expected = chi2_contingency(ct)
n = ct.values.sum()
r, k = ct.shape
cramers_v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
phi2 = chi2 / n
phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
cramers_v_corrected = np.sqrt(phi2corr / max(min(kcorr - 1, rcorr - 1), 1e-12))

# standardized residuals
resid = (ct.values - expected) / np.sqrt(np.maximum(expected, 1e-9))
resid_df = pd.DataFrame(resid, index=ct.index, columns=ct.columns)

# keep top products by frequency for readability
prod_top = df['product_id'].value_counts().head(20).index
resid_top = resid_df.loc[prod_top]

resid_df.to_csv(OUT / 'stage2_standardized_residuals_full.csv')
resid_top.to_csv(OUT / 'stage2_standardized_residuals_top20.csv')

plt.figure(figsize=(7,8))
plt.imshow(resid_top.values, cmap='coolwarm', aspect='auto', vmin=-np.max(np.abs(resid_top.values)), vmax=np.max(np.abs(resid_top.values)))
plt.colorbar(label='Std Residual')
plt.yticks(np.arange(len(resid_top.index)), resid_top.index)
plt.xticks(np.arange(len(resid_top.columns)), resid_top.columns)
plt.title('Standardized Residuals Heatmap (Top-20 Products)')
plt.tight_layout()
plt.savefig(OUT / 'stage2_heatmap_std_residuals_top20.png', dpi=180)
plt.close()

# -------------------- Stage 3: Divergence & Correspondence Analysis --------------------
# JS distance overall and by Need_Type
prod_all = sorted(df['product_id'].unique())

def mode_product_dist(subdf: pd.DataFrame, mode: str, products):
    vc = subdf[subdf['mode'] == mode]['product_id'].value_counts()
    arr = np.array([vc.get(p, 0) for p in products], dtype=float)
    if arr.sum() == 0:
        return np.zeros_like(arr)
    return arr / arr.sum()

p_v = mode_product_dist(df, 'visual', prod_all)
p_t = mode_product_dist(df, 'verbal', prod_all)
js_overall = float(jensenshannon(p_v, p_t, base=2))

js_by_need = {}
for need, sub in df.groupby('Need_Type'):
    prods = sorted(sub['product_id'].unique())
    pv = mode_product_dist(sub, 'visual', prods)
    pt = mode_product_dist(sub, 'verbal', prods)
    js_by_need[need] = float(jensenshannon(pv, pt, base=2))

# Correspondence Analysis (manual SVD)
# rows: condition (mode + need + expr), cols: product
ca_rows = df.assign(condition=df['mode'] + '|' + df['Need_Type'] + '|' + df['Expression_Type'])
N = pd.crosstab(ca_rows['condition'], ca_rows['product_id']).astype(float)
P = N / N.values.sum()
r_m = P.sum(axis=1).values.reshape(-1,1)
c_m = P.sum(axis=0).values.reshape(1,-1)
E = r_m @ c_m
S = (P.values - E) / np.sqrt(np.maximum(E, 1e-12))
U, s, VT = np.linalg.svd(S, full_matrices=False)

# principal coordinates
row_coord = (U[:, :2] * s[:2])
col_coord = (VT.T[:, :2] * s[:2])
row_df = pd.DataFrame(row_coord, index=N.index, columns=['Dim1','Dim2']).reset_index().rename(columns={'index':'condition'})
col_df = pd.DataFrame(col_coord, index=N.columns, columns=['Dim1','Dim2']).reset_index().rename(columns={'index':'product_id'})
row_df['mode'] = row_df['condition'].str.split('|').str[0]

row_df.to_csv(OUT / 'stage3_ca_row_coords.csv', index=False)
col_df.to_csv(OUT / 'stage3_ca_col_coords.csv', index=False)

plt.figure(figsize=(8,6))
for mode, sub in row_df.groupby('mode'):
    plt.scatter(sub['Dim1'], sub['Dim2'], label=mode, s=45, alpha=0.85)
for _, r0 in row_df.iterrows():
    plt.text(r0['Dim1'], r0['Dim2'], r0['condition'], fontsize=7, alpha=0.7)
plt.axhline(0, color='grey', lw=0.8)
plt.axvline(0, color='grey', lw=0.8)
plt.title('CA Biplot (Row Conditions: mode|Need|Expression)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'stage3_ca_biplot_rows.png', dpi=180)
plt.close()

# -------------------- Stage 4: Paired Validation --------------------
# pivot by run_id
paired = df.pivot_table(index=['run_id','category','code','Need_Type','Expression_Type'], columns='mode', values=['product_id','steps'], aggfunc='first').reset_index()
# flatten multi-index columns
paired.columns = ['_'.join([c for c in col if c]).strip('_') for col in paired.columns.to_flat_index()]

paired = paired.dropna(subset=['product_id_verbal','product_id_visual','steps_verbal','steps_visual'])
paired['same_product'] = paired['product_id_verbal'] == paired['product_id_visual']
paired['step_gap'] = paired['steps_verbal'] - paired['steps_visual']

divergence_rate = 1 - paired['same_product'].mean() if len(paired) else np.nan
immunity_rate = paired['same_product'].mean() if len(paired) else np.nan

paired.to_csv(OUT / 'stage4_paired_runs.csv', index=False)

# Dumbbell plot (top 40 for readability)
plot_df = paired.copy().sort_values('step_gap', ascending=False).head(40)
plot_df = plot_df.reset_index(drop=True)
plt.figure(figsize=(10, max(6, len(plot_df)*0.22)))
y = np.arange(len(plot_df))
for i, r0 in plot_df.iterrows():
    c = '#2ca02c' if r0['same_product'] else '#d62728'
    plt.plot([r0['steps_visual'], r0['steps_verbal']], [i, i], color=c, alpha=0.7)
plt.scatter(plot_df['steps_visual'], y, color='#1f77b4', label='Visual', s=28)
plt.scatter(plot_df['steps_verbal'], y, color='#ff7f0e', label='Verbal', s=28)
plt.yticks(y, plot_df['run_id'])
plt.xlabel('Steps')
plt.title('Dumbbell Plot: Visual vs Verbal Steps (Top 40 by gap)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'stage4_dumbbell_steps_top40.png', dpi=180)
plt.close()

# -------------------- Stage 5: Embedding-based Future Path (no embeddings yet) --------------------
stage5_notes = {
    'semantic_similarity': 'Use cosine similarity over product embeddings for matched/discordant recommendations.',
    'distribution_distance': 'Use Wasserstein distance between Visual and Verbal embedding distributions.',
    'nonlinear_projection': 'Use UMAP/t-SNE + KDE to visualize preference manifolds by mode.',
    'trajectory_shift': 'Track step-wise embedding path and compare convergence between modes.'
}

# -------------------- Save report --------------------
report = {
    'n_rows': int(len(df)),
    'n_run_pairs': int(len(paired)),
    'stage1': {
        'avg_steps_by_mode': stage1_mode.set_index('mode')['avg_steps'].to_dict(),
        'avg_steps_by_need_mode': stage1_need_mode.to_dict(orient='records'),
        'avg_steps_by_expr_mode': stage1_expr_mode.to_dict(orient='records'),
    },
    'stage2': {
        'cramers_v_product_need': float(cramers_v),
        'cramers_v_product_need_bias_corrected': float(cramers_v_corrected),
        'chi2_p_value': float(p),
        'contingency_shape': [int(r), int(k)],
    },
    'stage3': {
        'js_distance_overall': float(js_overall),
        'js_distance_by_need': js_by_need,
        'ca_singular_values_top5': [float(x) for x in s[:5]],
    },
    'stage4': {
        'divergence_rate': float(divergence_rate),
        'immunity_rate': float(immunity_rate),
        'avg_steps_verbal': float(paired['steps_verbal'].mean()) if len(paired) else None,
        'avg_steps_visual': float(paired['steps_visual'].mean()) if len(paired) else None,
    },
    'stage5': stage5_notes,
}

(OUT / 'analysis_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

# markdown summary
md = []
md.append('# Query 0317 Multi-Stage Analysis')
md.append(f'- Samples: {len(df)} recommendation records; paired runs: {len(paired)}')
md.append('')
md.append('## Stage 1: Descriptive & Efficiency')
for m, v in stage1_mode.set_index('mode')['avg_steps'].to_dict().items():
    md.append(f'- {m}: avg steps = {v:.3f}')
md.append('')
md.append('## Stage 2: Categorical Association')
md.append(f'- Cramér\'s V (product_id vs Need_Type): {cramers_v:.4f}')
md.append(f'- Bias-corrected Cramér\'s V: {cramers_v_corrected:.4f}')
md.append(f'- Chi-square p-value: {p:.4g}')
md.append('')
md.append('## Stage 3: Divergence & Projection')
md.append(f'- JS distance (Visual vs Verbal overall): {js_overall:.4f}')
for k0, v0 in js_by_need.items():
    md.append(f'- JS distance [{k0}]: {v0:.4f}')
md.append('')
md.append('## Stage 4: Paired Validation')
md.append(f'- Divergence rate (different products under same run): {divergence_rate*100:.2f}%')
md.append(f'- Immunity rate (same product under both modes): {immunity_rate*100:.2f}%')
if len(paired):
    md.append(f'- Avg steps verbal: {paired["steps_verbal"].mean():.3f}')
    md.append(f'- Avg steps visual: {paired["steps_visual"].mean():.3f}')
md.append('')
md.append('## Stage 5: Embedding-based Evolution')
for k0, v0 in stage5_notes.items():
    md.append(f'- {k0}: {v0}')

(OUT / 'analysis_summary.md').write_text('\n'.join(md), encoding='utf-8')

print(json.dumps(report, ensure_ascii=False, indent=2))
print(f'Artifacts saved to: {OUT}')
