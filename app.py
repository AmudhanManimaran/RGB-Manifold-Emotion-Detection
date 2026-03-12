import os
import uuid
import csv
import math
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_labels():
    labels = []
    with open('data/labels.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rgb = (int(row['R']), int(row['G']), int(row['B']))
            labels.append((rgb, row['Emotion']))
    return labels

emotion_labels = load_labels()

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def rgb_to_hsi(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    intensity = (r + g + b) / 3.0
    min_val = min(r, g, b)
    saturation = 0 if intensity == 0 else 1 - min_val / intensity
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = math.sqrt((r - g)**2 + (r - b)*(g - b)) + 1e-6
    theta = math.acos(numerator / denominator)
    hue = theta if b <= g else (2 * math.pi - theta)
    hue_deg = round(math.degrees(hue))
    return round(hue_deg), round(saturation, 2), round(intensity, 2)

def match_emotion(color_rgb, tolerance=0):
    closest_emotion = None
    closest_distance = float('inf')
    for label_rgb, emotion in emotion_labels:
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(color_rgb, label_rgb)))
        if distance < closest_distance:
            closest_distance = distance
            closest_emotion = emotion

    if closest_distance < tolerance:
        confidence = 100.0
    else:
        max_dist = math.sqrt(255 ** 2 * 3)
        confidence = round((1 - (closest_distance / max_dist)) * 100, 2)

    return closest_emotion, confidence

def get_dominant_colors_with_weights(image, k=25, distance_threshold=0, method='kmeans'):
    image = image.resize((250, 250), Image.NEAREST)
    img_data = np.array(image).reshape(-1, 3)
    unique_colors = np.unique(img_data, axis=0)

    if len(unique_colors) == 1:
        return [tuple(unique_colors[0])], [100.0]

    actual_k = min(k, len(unique_colors))

    if method == 'gmm':
        model = GaussianMixture(n_components=actual_k, covariance_type='tied', random_state=42)
    else:
        model = KMeans(n_clusters=actual_k, n_init=10, random_state=42)

    model.fit(img_data)

    labels = model.predict(img_data)
    _, counts = np.unique(labels, return_counts=True)
    total = sum(counts)

    if method == 'gmm':
        raw_colors = [tuple(map(int, model.means_[i])) for i in range(actual_k)]
    else:
        raw_colors = [tuple(map(int, model.cluster_centers_[i])) for i in range(actual_k)]

    raw_weights = [round((count / total) * 100, 2) for count in counts]

    sorted_by_weight = sorted(zip(raw_colors, raw_weights), key=lambda x: -x[1])
    filtered_colors, filtered_weights = [], []
    for color, weight in sorted_by_weight:
        if all(math.dist(color, c) > distance_threshold for c in filtered_colors):
            filtered_colors.append(color)
            filtered_weights.append(weight)

    return filtered_colors, filtered_weights

@app.route('/')
def upload_form():
    return render_template('upload.html', title='Upload an Image')

@app.route('/gallery')
def gallery():
    return "Gallery Page Coming Soon"

@app.route('/hsi-reference')
def hsi_reference():
    return render_template('hsi-reference.html', title='HSI Reference Table')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('upload_form'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('upload_form'))

    clustering_method = request.form.get('method', 'kmeans')  
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    resized_image = Image.open(filepath).convert('RGB')
    dominant_colors, color_weights = get_dominant_colors_with_weights(
        resized_image,
        k=25,
        method=clustering_method
    )

    color_emotion_map = []
    
    # --- ALTERATION 1: Added a second counter for Prevalence Weights ---
    emotions_counter = Counter()
    emotions_prevalence = Counter() 
    
    hues, saturations, intensities = [], [], []
    total_weighted_accuracy = 0

    for color, weight in zip(dominant_colors, color_weights):
        emotion, confidence = match_emotion(color)
        
        # --- ALTERATION 2: Track both Count (+1) and Prevalence (+weight) ---
        emotions_counter[emotion] += 1
        emotions_prevalence[emotion] += weight
        
        h, s, i = rgb_to_hsi(color)
        hues.append(h)
        saturations.append(s)
        intensities.append(i)
        total_weighted_accuracy += (confidence * weight) / 100
        color_emotion_map.append((
            rgb_to_hex(color),
            f"({color[0]}, {color[1]}, {color[2]})",
            emotion,
            confidence
        ))

    resized_accuracy = round(total_weighted_accuracy, 2)

    def save_weighted_color_chart(colors, weights, filename):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(colors)), weights,
               color=[np.array(c) / 255 for c in colors],
               tick_label=[rgb_to_hex(c) for c in colors])
        ax.set_title('Dominant Colors by Pixel Percentage')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Color (Hex)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        plt.savefig(save_path)
        plt.close()
        return filename

    color_chart = save_weighted_color_chart(dominant_colors, color_weights, 'color_chart.png')

    # The chart remains based on counts as per your original logic
    emotion_chart = os.path.join(app.config['UPLOAD_FOLDER'], 'emotion_chart.png')
    plt.figure()
    plt.bar(emotions_counter.keys(), emotions_counter.values(), color='skyblue')
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(emotion_chart)
    plt.close()

    hue_pie_chart = os.path.join(app.config['UPLOAD_FOLDER'], 'hue_pie_chart.png')
    plt.figure()
    plt.pie([1] * len(hues), labels=hues, colors=[rgb_to_hex(c) for c in dominant_colors])
    plt.title("Hue Distribution")
    plt.savefig(hue_pie_chart)
    plt.close()

    saturation_bar_chart = os.path.join(app.config['UPLOAD_FOLDER'], 'saturation_bar_chart.png')
    plt.figure()
    plt.bar(range(len(saturations)), saturations, color='green')
    plt.title("Saturation Levels")
    plt.savefig(saturation_bar_chart)
    plt.close()

    intensity_bar_chart = os.path.join(app.config['UPLOAD_FOLDER'], 'intensity_bar_chart.png')
    plt.figure()
    plt.bar(range(len(intensities)), intensities, color='orange')
    plt.title("Intensity Levels")
    plt.savefig(intensity_bar_chart)
    plt.close()

    color_table = [
        {
            'hex': rgb_to_hex(color),
            'rgb': f"({color[0]}, {color[1]}, {color[2]})",
            'percentage': f"{weight:.2f}%"
        }
        for color, weight in zip(dominant_colors, color_weights)
    ]

    return render_template('result.html',
                           title='Emotion Results',
                           uploaded_image=filename,
                           color_chart=os.path.basename(color_chart),
                           emotion_chart=os.path.basename(emotion_chart),
                           hue_pie_chart=os.path.basename(hue_pie_chart),
                           saturation_bar_chart=os.path.basename(saturation_bar_chart),
                           intensity_bar_chart=os.path.basename(intensity_bar_chart),
                           color_emotion_map=color_emotion_map,
                           color_table=color_table,
                           emotions=dict(emotions_counter),
                           # --- ALTERATION 3: Pass the new prevalence weights to the template ---
                           emotions_prevalence={k: round(v, 2) for k, v in emotions_prevalence.items()},
                           hues=hues,
                           saturations=saturations,
                           intensities=intensities,
                           accuracy=resized_accuracy)

if __name__ == '__main__':
    app.run(debug=True)