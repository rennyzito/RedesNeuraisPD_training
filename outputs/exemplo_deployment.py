from joblib import load
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar modelo treinado
model = load('melhor_modelo.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predições"""
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(probability),
            'team_win_probability': {
                'blue': float(model.predict_proba(features)[0][1]),
                'red': float(model.predict_proba(features)[0][0])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
