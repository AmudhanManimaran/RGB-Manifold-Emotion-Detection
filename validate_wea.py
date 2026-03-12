import os
import csv
import math
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_rel
import pandas as pd

# 1. Define the specific emotion names for Config A (8) and Config B (24)
CONFIG_A_EMOTIONS = [
    'Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation'
]

CONFIG_B_EMOTIONS = CONFIG_A_EMOTIONS + [
    'Serenity', 'Ecstasy', 'Acceptance', 'Admiration', 'Apprehension', 'Terror', 
    'Distraction', 'Amazement', 'Pensiveness', 'Grief', 'Boredom', 'Loathing', 
    'Annoyance', 'Rage', 'Interest', 'Vigilance'
]

# 2. Load Anchors and Generate Random Baseline
def load_anchor_configs():
    anchors_A, anchors_B, anchors_C = [], [], []
    with open('data/labels.csv', 'r') as file:
        lines = (line for line in file if line.strip())
        reader = csv.DictReader(lines)
        
        for row in reader:
            emotion_name = row['Emotion'].strip()
            rgb = (int(row['R']), int(row['G']), int(row['B']))
            
            anchors_C.append(rgb)
            if emotion_name in CONFIG_B_EMOTIONS:
                anchors_B.append(rgb)
            if emotion_name in CONFIG_A_EMOTIONS:
                anchors_A.append(rgb)
                
    print(f"Loaded: Config A ({len(anchors_A)}), Config B ({len(anchors_B)}), Config C ({len(anchors_C)})")
    return anchors_A, anchors_B, anchors_C

def generate_random_anchors(n=48, seed=42):
    """Generates a purely random 48-anchor RGB baseline for comparison."""
    np.random.seed(seed)
    return [tuple(rgb) for rgb in np.random.randint(0, 256, size=(n, 3))]

# 3. Feature Extraction (Strictly K=25)
def extract_clusters(image_path, k=25):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((250, 250), Image.NEAREST)
    img_data = np.array(image).reshape(-1, 3)
    
    unique_colors = np.unique(img_data, axis=0)
    actual_k = min(k, len(unique_colors))
    
    model = GaussianMixture(n_components=actual_k, covariance_type='tied', random_state=42)
    model.fit(img_data)
    labels = model.predict(img_data)
    
    _, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    
    colors = [tuple(map(int, model.means_[i])) for i in range(actual_k)]
    weights = [count / total for count in counts]
    
    return colors, weights

# 4. WEA Math
def calculate_wea(colors, weights, anchors):
    wea_score = 0
    max_dist = math.sqrt(255**2 * 3)
    for cluster_color, weight in zip(colors, weights):
        min_dist = min(math.dist(cluster_color, anchor) for anchor in anchors)
        mc_i = 1.0 - (min_dist / max_dist) 
        wea_score += (weight * mc_i)
    return wea_score

# Helper for Statistics
def get_stats(list_target, list_baseline):
    t_stat, p_value = ttest_rel(list_target, list_baseline)
    diff = np.array(list_target) - np.array(list_baseline)
    std_diff = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0
    return round(cohens_d, 2), "< 0.001" if p_value < 0.001 else round(p_value, 3)

# 5. Main Unified Validation Loop
def run_unified_validation():
    anchors_A, anchors_B, anchors_C = load_anchor_configs()
    anchors_Random = generate_random_anchors(n=48)
    
    dataset_dir = 'dataset'
    styles = ['Cubism', 'Impressionism', 'Nihonga', 'Romanticism']
    
    ablation_results = []
    statistical_results = []
    
    # Global/Pooled tracking
    all_A, all_B, all_C, all_R = [], [], [], []
    
    print("\nStarting Unified Academic Validation Pipeline...")
    for style in styles:
        print(f"Processing style: {style}...")
        style_dir = os.path.join(dataset_dir, style)
        
        if not os.path.exists(style_dir):
            print(f"  -> Directory {style_dir} not found. Skipping.")
            continue
            
        wea_A_list, wea_B_list, wea_C_list, wea_R_list = [], [], [], []
        
        for filename in os.listdir(style_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(style_dir, filename)
                colors, weights = extract_clusters(img_path, k=25)
                
                # Evaluate against all 4 configurations
                wea_A_list.append(calculate_wea(colors, weights, anchors_A))
                wea_B_list.append(calculate_wea(colors, weights, anchors_B))
                wea_C_list.append(calculate_wea(colors, weights, anchors_C))
                wea_R_list.append(calculate_wea(colors, weights, anchors_Random))
        
        # Extend global pooled lists
        all_A.extend(wea_A_list)
        all_B.extend(wea_B_list)
        all_C.extend(wea_C_list)
        all_R.extend(wea_R_list)
        
        if len(wea_C_list) > 0:
            # Table 1 Entry (Means ± Variance)
            ablation_results.append({
                'Style': style,
                'Samples': len(wea_C_list),
                'Random (48)': f"{np.mean(wea_R_list):.3f} ± {np.std(wea_R_list):.3f}",
                'Config A (8)': f"{np.mean(wea_A_list):.3f} ± {np.std(wea_A_list):.3f}",
                'Config B (24)': f"{np.mean(wea_B_list):.3f} ± {np.std(wea_B_list):.3f}",
                'Config C (48)': f"{np.mean(wea_C_list):.3f} ± {np.std(wea_C_list):.3f}",
                'Total Gain (A to C)': f"+{(np.mean(wea_C_list) - np.mean(wea_A_list)):.3f}"
            })
            
            # Table 2 Entry (Pairwise Significance)
            d_AB, p_AB = get_stats(wea_B_list, wea_A_list)
            d_BC, p_BC = get_stats(wea_C_list, wea_B_list)
            d_AC, p_AC = get_stats(wea_C_list, wea_A_list)
            
            statistical_results.append({
                'Style': style,
                'A vs B (d)': d_AB, 'A vs B (p)': p_AB,
                'B vs C (d)': d_BC, 'B vs C (p)': p_BC,
                'A vs C (d)': d_AC, 'A vs C (p)': p_AC
            })

    # Process OVERALL (Pooled) Results
    if all_C:
        ablation_results.append({
            'Style': 'OVERALL',
            'Samples': len(all_C),
            'Random (48)': f"{np.mean(all_R):.3f} ± {np.std(all_R):.3f}",
            'Config A (8)': f"{np.mean(all_A):.3f} ± {np.std(all_A):.3f}",
            'Config B (24)': f"{np.mean(all_B):.3f} ± {np.std(all_B):.3f}",
            'Config C (48)': f"{np.mean(all_C):.3f} ± {np.std(all_C):.3f}",
            'Total Gain (A to C)': f"+{(np.mean(all_C) - np.mean(all_A)):.3f}"
        })
        
        d_AB, p_AB = get_stats(all_B, all_A)
        d_BC, p_BC = get_stats(all_C, all_B)
        d_AC, p_AC = get_stats(all_C, all_A)
        
        statistical_results.append({
            'Style': 'OVERALL',
            'A vs B (d)': d_AB, 'A vs B (p)': p_AB,
            'B vs C (d)': d_BC, 'B vs C (p)': p_BC,
            'A vs C (d)': d_AC, 'A vs C (p)': p_AC
        })

    # Print clean tables
    df_ablation = pd.DataFrame(ablation_results)
    df_stats = pd.DataFrame(statistical_results)
    
    df_ablation.to_csv('ablation_table.csv', index=False)
    df_stats.to_csv('statistical_table.csv', index=False)
    
    print("\n" + "="*85)
    print("TABLE 1: ABLATION STUDY (Mean WEA ± Std Dev)")
    print("="*85)
    print(df_ablation.to_string(index=False))
    
    print("\n" + "="*85)
    print("TABLE 2: LAYER-WISE STATISTICAL SIGNIFICANCE (Cohen's d and p-values)")
    print("="*85)
    print(df_stats.to_string(index=False))
    
    print("\nResults saved to 'ablation_table.csv' and 'statistical_table.csv'.")

if __name__ == "__main__":
    run_unified_validation()