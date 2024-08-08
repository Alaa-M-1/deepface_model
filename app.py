from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)

def analyze_facial_expressions(image):
    try:
        # Convert the image to an OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_path = 'temp.jpg'
        cv2.imwrite(image_path, image)
        
        # Analyze the image for facial expressions
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])

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

                # Prepare to draw the text on the image
                cv2.putText(image, f"Dominant emotion: {dominant_emotion} ({dominant_prob:.1f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Check if the second highest emotion probability is above 40%
                if len(sorted_emotions) > 1:
                    second_emotion, second_prob = sorted_emotions[1]
                    if second_prob > 40:
                        cv2.putText(image, f"Second emotion: {second_emotion} ({second_prob:.1f}%)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Encode image to base64
                _, buffer = cv2.imencode('.jpg', image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                return {'dominant_emotion': dominant_emotion, 'dominant_prob': dominant_prob, 'image_base64': image_base64}

            else:
                return {'error': 'Unexpected result format.'}
        else:
            return {'error': 'No results returned.'}

    except Exception as e:
        return {'error': str(e)}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    result = analyze_facial_expressions(image)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
