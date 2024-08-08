from flask import Flask, render_template, request
from deepface import DeepFace
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)

def analyze_facial_expressions(image):
    try:
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)

        # Analyze the image for facial expressions directly from the NumPy array
        result = DeepFace.analyze(img_path=image_array, actions=['emotion'])

        # Extract the first result
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if 'dominant_emotion' in first_result and 'emotion' in first_result:
                # Extract and sort emotions by probability
                emotions = first_result['emotion']
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

                # Get dominant emotion and probability
                dominant_emotion = first_result['dominant_emotion']
                dominant_prob = emotions[dominant_emotion]

                # Initialize variables to store the second emotion and its probability
                second_emotion = None
                second_prob = None

                # Check if the second highest emotion probability is above 40%
                if len(sorted_emotions) > 1:
                    second_emotion, second_prob = sorted_emotions[1]
                    if second_prob <= 40:
                        second_emotion = None
                        second_prob = None

                # Convert image to base64 for display
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                return {
                    'dominant_emotion': dominant_emotion,
                    'dominant_prob': dominant_prob,
                    'second_emotion': second_emotion,
                    'second_prob': second_prob,
                    'image_base64': img_base64
                }

            else:
                return None
        else:
            return None

    except Exception as e:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file provided.')

        file = request.files['file']
        image = Image.open(file.stream)
        result = analyze_facial_expressions(image)

        if result:
            return render_template(
                'index.html', 
                dominant_emotion=result['dominant_emotion'], 
                dominant_prob=result['dominant_prob'], 
                second_emotion=result['second_emotion'], 
                second_prob=result['second_prob'], 
                image_base64=result['image_base64']
            )
        else:
            return render_template('index.html', error='Error in processing the image.')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
