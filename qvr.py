"""
Fig 7 — Qualitative Visual Results: Artwork Grid
Plutchik-48: GMM Color Palette Extraction + Emotion Prediction
Journal-quality figure for The Visual Computer — TEXT OVERLAP FIXED VERSION

PIPELINE LOGIC: Unchanged from app.py
VISUALIZATION:  Complete professional overhaul, all text overlap resolved
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from PIL import Image
import colorsys
import math
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# 1. PLUTCHIK-48 GROUND TRUTH  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
plutchik_48 = {
    'Anger': (220, 0, 0), 'Anticipation': (255, 155, 0), 'Joy': (255, 255, 0), 'Trust': (0, 200, 0),
    'Fear': (100, 56, 135), 'Surprise': (0, 170, 138), 'Sadness': (70, 100, 173), 'Disgust': (128, 128, 0),
    'Rage': (150, 0, 0), 'Vigilance': (173, 105, 0), 'Ecstasy': (173, 173, 0), 'Admiration': (0, 136, 0),
    'Terror': (68, 38, 92), 'Amazement': (0, 116, 94), 'Grief': (48, 68, 118), 'Loathing': (87, 87, 0),
    'Annoyance': (241, 153, 153), 'Interest': (255, 215, 153), 'Serenity': (255, 255, 153), 'Acceptance': (153, 233, 153),
    'Apprehension': (193, 175, 207), 'Distraction': (153, 221, 208), 'Pensiveness': (181, 193, 222), 'Boredom': (204, 204, 153),
    'Love': (128, 228, 0), 'Submission': (50, 128, 68), 'Awe': (50, 113, 136), 'Disapproval': (35, 135, 156),
    'Remorse': (99, 114, 86), 'Contempt': (174, 64, 0), 'Aggressiveness': (238, 78, 0), 'Optimism': (255, 205, 0),
    'Pride': (238, 128, 0), 'Hope': (128, 178, 0), 'Guilt': (178, 156, 86), 'Curiosity': (0, 185, 69),
    'Despair': (85, 78, 154), 'Unbelief': (64, 149, 69), 'Envy': (145, 50, 86), 'Cynicism': (192, 142, 0),
    'Outrage': (110, 85, 69), 'Pessimism': (162, 127, 86), 'Morbidity': (192, 192, 0), 'Dominance': (110, 100, 0),
    'Anxiety': (178, 106, 68), 'Delight': (128, 213, 69), 'Sentimentality': (35, 150, 85), 'Shame': (114, 92, 67)
}

# ── Emotion bar colors ────────────────────────────────────────────────────────
TIER_COLOR = {
    'Anger':'#C00000','Anticipation':'#E07800','Joy':'#B8B800','Trust':'#006600',
    'Fear':'#3D1A5A','Surprise':'#007060','Sadness':'#2A3D7A','Disgust':'#505000',
    'Rage':'#8B0000','Vigilance':'#8B5500','Ecstasy':'#8B8B00','Admiration':'#004400',
    'Terror':'#2A1040','Amazement':'#004A3C','Grief':'#1A2848','Loathing':'#3A3A00',
    'Annoyance':'#C06060','Interest':'#C08040','Serenity':'#C0C060','Acceptance':'#60A060',
    'Apprehension':'#7060A0','Distraction':'#409090','Pensiveness':'#607090','Boredom':'#808040',
    'Love':'#507800','Submission':'#204830','Awe':'#204858','Disapproval':'#205868',
    'Remorse':'#404830','Contempt':'#883000','Aggressiveness':'#B83000','Optimism':'#C08000',
    'Pride':'#B06000','Hope':'#507000','Guilt':'#806840','Curiosity':'#007040',
    'Despair':'#352060','Unbelief':'#205830','Envy':'#602038','Cynicism':'#806000',
    'Outrage':'#484030','Pessimism':'#604838','Morbidity':'#787800','Dominance':'#484000',
    'Anxiety':'#804030','Delight':'#508830','Sentimentality':'#106038','Shame':'#484030',
}

def get_tier(emotion):
    if emotion in {'Anger','Anticipation','Joy','Trust','Fear','Surprise','Sadness','Disgust'}:
        return 'Primary'
    if emotion in {'Rage','Vigilance','Ecstasy','Admiration','Terror','Amazement','Grief','Loathing'}:
        return 'High-Intensity'
    if emotion in {'Annoyance','Interest','Serenity','Acceptance','Apprehension','Distraction','Pensiveness','Boredom'}:
        return 'Low-Intensity'
    return 'Dyadic'

TIER_BADGE_COL = {
    'Primary':        '#1A1A2E',
    'High-Intensity': '#8B0000',
    'Low-Intensity':  '#0F3460',
    'Dyadic':         '#533483',
}

# ── Style accent colors ───────────────────────────────────────────────────────
STYLE_COLORS = {
    'Cubism':        '#1B4F72',
    'Impressionism': '#0E6655',
    'Nihonga':       '#145A32',
    'Romanticism':   '#641E16',
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. PIPELINE FUNCTIONS  (logic unchanged from app.py)
# ══════════════════════════════════════════════════════════════════════════════
def get_dominant_colors_with_weights(image_path, k=25, distance_threshold=0):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((250, 250), Image.NEAREST)
    img_data = np.array(image).reshape(-1, 3)
    unique_colors = np.unique(img_data, axis=0)
    if len(unique_colors) == 1:
        return [tuple(unique_colors[0])], [100.0]
    actual_k = min(k, len(unique_colors))
    model = GaussianMixture(n_components=actual_k, covariance_type='tied', random_state=42)
    model.fit(img_data)
    labels = model.predict(img_data)
    _, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    raw_colors  = [tuple(map(int, model.means_[i])) for i in range(actual_k)]
    raw_weights = [round((count / total) * 100, 2) for count in counts]
    sorted_by_weight = sorted(zip(raw_colors, raw_weights), key=lambda x: -x[1])
    filtered_colors, filtered_weights = [], []
    for color, weight in sorted_by_weight:
        if all(math.dist(color, c) > distance_threshold for c in filtered_colors):
            filtered_colors.append(color)
            filtered_weights.append(weight)
    return filtered_colors, filtered_weights

def calculate_top3_prevalence(colors, weights):
    emotions_prevalence = Counter()
    for color, weight in zip(colors, weights):
        min_dist = float('inf')
        closest_emotion = None
        for emotion, p_rgb in plutchik_48.items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(color, p_rgb)))
            if dist < min_dist:
                min_dist = dist
                closest_emotion = emotion
        emotions_prevalence[closest_emotion] += weight
    top_3 = sorted(emotions_prevalence.items(), key=lambda x: x[1], reverse=True)[:3]
    return dict(top_3)

def sort_palette_by_hue(colors):
    def key(c):
        h, s, v = colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)
        return (round(h * 12) / 12, v)
    return sorted(colors, key=key)

# ══════════════════════════════════════════════════════════════════════════════
# 3. ARTWORK INPUT
# ══════════════════════════════════════════════════════════════════════════════
artworks = [
    {'path': 'C_11.png',  'style': 'Cubism'},
    {'path': 'Im_13.png', 'style': 'Impressionism'},
    {'path': 'N_11.png',  'style': 'Nihonga'},
    {'path': 'R_3.png',   'style': 'Romanticism'},
]

results = []
for art in artworks:
    try:
        print(f"  Processing {art['style']}...")
        colors, weights = get_dominant_colors_with_weights(art['path'])
        emotions = calculate_top3_prevalence(colors, weights)
        results.append({
            'style':      art['style'],
            'image_path': art['path'],
            'colors':     colors,
            'weights':    weights,
            'emotions':   emotions,
        })
        print(f"    ✓ {art['style']} → {list(emotions.keys())[0]}")
    except FileNotFoundError:
        print(f"  ✗ '{art['path']}' not found.")

if len(results) != 4:
    print(f"\nOnly {len(results)}/4 loaded. Check paths.")
    exit()

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE — 2 rows × 2 columns layout
#    Each cell: [image | palette strip | bar chart] stacked vertically
#    All spacing calculated explicitly — no overlap possible
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':      'DejaVu Serif',
    'font.size':        10,
    'axes.linewidth':   0.8,
    'figure.facecolor': 'white',
})

# ── Master figure: 2×2 grid of artwork panels ─────────────────────────────────
fig = plt.figure(figsize=(18, 22), facecolor='white', dpi=300)

# Outer 2×2 grid — each cell is one artwork
outer = gridspec.GridSpec(
    2, 2,
    figure=fig,
    left=0.05, right=0.97,
    top=0.97, bottom=0.06,
    wspace=0.10,
    hspace=0.12,
)

# ── Draw each artwork panel ───────────────────────────────────────────────────
for idx, data in enumerate(results):
    row = idx // 2
    col = idx % 2

    style       = data['style']
    emotions    = data['emotions']
    colors      = data['colors']
    top_emotion = list(emotions.keys())[0]
    top_pct     = list(emotions.values())[0]
    tier        = get_tier(top_emotion)
    sc          = STYLE_COLORS[style]
    ec          = TIER_COLOR.get(top_emotion, '#333333')

    # Each cell: image (top) + palette strip (middle) + bar chart (bottom)
    cell = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[row, col],
        height_ratios=[10, 0.55, 5.5],
        hspace=0.0,
    )

    # ── PANEL A: Artwork Image ────────────────────────────────────────────────
    ax_img = fig.add_subplot(cell[0])
    img    = Image.open(data['image_path']).convert('RGB')
    img_w, img_h = img.size
    ax_img.imshow(img)
    ax_img.set_aspect(img_h / img_w)   # lock axes to image's natural ratio
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # Colored border
    for sp in ax_img.spines.values():
        sp.set_edgecolor(sc); sp.set_linewidth(3.0)

    # ── Style name — TOP CENTRE, ABOVE image (not on image) ──────────────────
    ax_img.set_title(
        style,
        fontsize=15, fontweight='bold', color='white',
        pad=0,
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor=sc, edgecolor='white',
                  linewidth=1.5, alpha=0.96)
    )

    # ── Tier badge — top-right INSIDE image, small ────────────────────────────
    ax_img.text(
        0.985, 0.985, tier,
        transform=ax_img.transAxes,
        ha='right', va='top',
        fontsize=8.5, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor=TIER_BADGE_COL[tier],
                  edgecolor='white', linewidth=0.8, alpha=0.90)
    )

    # ── Predicted emotion — BOTTOM of image, separate full-width bar ─────────
    # Use a thin axes BELOW the image but above palette for clean separation
    ax_img.text(
        0.5, -0.025,
        f'Predicted Emotion:   {top_emotion}   ({top_pct:.1f}%)',
        transform=ax_img.transAxes,
        ha='center', va='top',
        fontsize=11.5, fontweight='bold', color='white',
        clip_on=False,
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor=ec,
            edgecolor=sc, linewidth=1.5,
            alpha=0.97
        )
    )

    # ── PANEL B: GMM Palette Strip ────────────────────────────────────────────
    ax_pal = fig.add_subplot(cell[1])
    sorted_colors = sort_palette_by_hue(colors)
    palette_arr   = np.array(sorted_colors, dtype=np.uint8).reshape(1, -1, 3)
    ax_pal.imshow(palette_arr, aspect='auto', interpolation='nearest')
    ax_pal.set_xticks([]); ax_pal.set_yticks([])
    for sp in ax_pal.spines.values():
        sp.set_edgecolor(sc); sp.set_linewidth(1.8)

    # Palette label — placed BELOW the strip as xlabel
    ax_pal.set_xlabel(
        f'GMM Dominant Palette  ·  K = {len(colors)} components  ·  sorted by hue',
        fontsize=9.0, color='#444444', style='italic', labelpad=5
    )

    # ── PANEL C: Top-3 Prevalence Bar Chart ───────────────────────────────────
    ax_bar = fig.add_subplot(cell[2])
    ax_bar.set_facecolor('#F8FAFC')

    emo_names  = list(emotions.keys())[::-1]      # bottom → top order
    pcts       = list(emotions.values())[::-1]
    bar_cols   = [TIER_COLOR.get(e, '#555555') for e in emo_names]

    bars = ax_bar.barh(
        range(len(emo_names)), pcts,
        color=bar_cols,
        edgecolor='white',
        linewidth=1.0,
        height=0.50,
        zorder=3,
    )

    # Highlight the top prediction bar
    bars[-1].set_edgecolor(sc)
    bars[-1].set_linewidth(2.2)

    # y-tick labels: emotion names — set explicitly, no overlap with values
    ax_bar.set_yticks(range(len(emo_names)))
    ax_bar.set_yticklabels(emo_names, fontsize=12.0, fontweight='normal')
    ax_bar.tick_params(axis='y', length=0, pad=6)

    # Value labels at END of bars — pushed far right with fixed offset
    max_val = max(pcts) if pcts else 100
    ax_bar.set_xlim(0, max_val * 1.40)

    for bar_obj, pct, emo in zip(bars, pcts, emo_names):
        ax_bar.text(
            pct + max_val * 0.04,
            bar_obj.get_y() + bar_obj.get_height() / 2.0,
            f'{pct:.1f}%',
            va='center', ha='left',
            fontsize=12.0, fontweight='bold',
            color=TIER_COLOR.get(emo, '#222222')
        )

    # Reference gridlines (subtle)
    ax_bar.xaxis.grid(True, linestyle=':', linewidth=0.6,
                      color='#CCCCCC', alpha=0.7, zorder=0)
    ax_bar.set_axisbelow(True)

    # Clean axes
    ax_bar.set_xticks([])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.spines['left'].set_color('#DDDDDD')

    # x-axis label
    ax_bar.set_xlabel(
        'Weighted Prevalence  (%)',
        fontsize=9.0, color='#777777', labelpad=5
    )

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE-LEVEL ELEMENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Tier legend — bottom centre ───────────────────────────────────────────────
tier_handles = [
    mpatches.Patch(facecolor=TIER_BADGE_COL['Primary'],        edgecolor='white',
                   label='Primary Anchor'),
    mpatches.Patch(facecolor=TIER_BADGE_COL['High-Intensity'], edgecolor='white',
                   label='High-Intensity Derivative'),
    mpatches.Patch(facecolor=TIER_BADGE_COL['Low-Intensity'],  edgecolor='white',
                   label='Low-Intensity Derivative'),
    mpatches.Patch(facecolor=TIER_BADGE_COL['Dyadic'],         edgecolor='white',
                   label='Dyadic Combination'),
]
fig.legend(
    handles=tier_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.008),
    ncol=4,
    fontsize=10,
    frameon=True, framealpha=0.95,
    edgecolor='#BBBBBB',
    title='Plutchik-48 Derivation Tier of Predicted Emotion',
    title_fontsize=10,
    columnspacing=2.0,
    handlelength=1.5,
)



# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_png = 'Qualitative_Visual_Results_Final.png'
out_pdf = 'Qualitative_Visual_Results_Final.pdf'

plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓  {out_png}")
print(f"✓  {out_pdf}")