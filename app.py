from flask import Flask, request, jsonify
import numpy as np
import logging
from flask import make_response

# Try to import flask-cors (optional) and sklearn; fall back when unavailable
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

# Try to import sklearn; if unavailable, fall back to a lightweight heuristic model
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    SKLEARN_AVAILABLE = False

app = Flask(__name__, static_folder='static', static_url_path='')
if CORS_AVAILABLE:
    CORS(app)
    logging.info('Enabled CORS (flask-cors available).')
else:
    logging.info('flask-cors not installed; requests from other origins may be blocked.')
logging.basicConfig(level=logging.INFO)

# Labels
LABELS = {0: 'No Snow', 1: 'Light Snow', 2: 'Heavy Snow'}


def generate_synthetic(n=1000, random_state=0):
    rng = np.random.RandomState(random_state)
    temps = rng.uniform(-20, 10, size=n)  # Celsius
    hums = rng.uniform(10, 100, size=n)
    X = np.vstack([temps, hums]).T

    y = []
    for t, h in X:
        if t <= 0 and h > 70:
            y.append(2)
        elif t <= 2 and h > 50:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    # add a bit of label noise
    flip = rng.rand(n) < 0.05
    y[flip] = rng.randint(0, 3, size=flip.sum())
    return X, y


class HeuristicModel:
    """Fallback model when scikit-learn isn't available."""
    def predict(self, X):
        preds = []
        for row in X:
            t, h = float(row[0]), float(row[1])
            if t <= 0 and h > 70:
                preds.append(2)
            elif t <= 2 and h > 50:
                preds.append(1)
            else:
                preds.append(0)
        return np.array(preds)

    def predict_proba(self, X):
        probs = []
        for row in X:
            t, h = float(row[0]), float(row[1])
            if t <= 0 and h > 70:
                probs.append([0.05, 0.2, 0.75])
            elif t <= 2 and h > 50:
                probs.append([0.1, 0.7, 0.2])
            else:
                probs.append([0.8, 0.15, 0.05])
        return np.array(probs)


logging.info('Preparing model...')
if SKLEARN_AVAILABLE:
    X, y = generate_synthetic(2000)
    model = RandomForestClassifier(n_estimators=50, random_state=1)
    model.fit(X, y)
    logging.info('Trained RandomForest model (scikit-learn available).')
else:
    model = HeuristicModel()
    logging.info('scikit-learn not available; using HeuristicModel fallback.')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
            temp = float(data.get('temperature', 0))
            hum = float(data.get('humidity', 50))
        else:
            temp = float(request.args.get('temperature', 0))
            hum = float(request.args.get('humidity', 50))

        Xq = np.array([[temp, hum]])
        pred = int(model.predict(Xq)[0])
        probs = model.predict_proba(Xq)[0].tolist()
        return jsonify({
            'prediction': pred,
            'label': LABELS.get(pred, 'Unknown'),
            'probabilities': probs,
            'input': {'temperature': temp, 'humidity': hum}
        })
    except Exception as e:
        logging.exception('Prediction error')
        return jsonify({'error': 'Invalid input or server error', 'detail': str(e)}), 400


if __name__ == '__main__':
    # Bind to all interfaces so localhost and other dev tools can reach it; keep debug on for dev
    app.run(host='0.0.0.0', port=5000, debug=True)
