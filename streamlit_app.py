import streamlit as st
import numpy as np
import logging
import os

# Try sklearn first, otherwise use heuristic
try:
    from sklearn.ensemble import RandomForestClassifier
    from joblib import dump, load
    SKLEARN = True
except Exception:
    RandomForestClassifier = None
    SKLEARN = False

st.set_page_config(page_title='Winter AI — Snowiness', layout='centered')
logging.basicConfig(level=logging.INFO)

LABELS = {0: 'No Snow', 1: 'Light Snow', 2: 'Heavy Snow'}

@st.cache_data
def generate_synthetic(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    temps = rng.uniform(-20, 10, size=n)
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
    flip = rng.rand(n) < 0.05
    y[flip] = rng.randint(0, 3, size=flip.sum())
    return X, y

class HeuristicModel:
    def predict(self, X):
        res = []
        for row in X:
            t, h = float(row[0]), float(row[1])
            if t <= 0 and h > 70:
                res.append(2)
            elif t <= 2 and h > 50:
                res.append(1)
            else:
                res.append(0)
        return np.array(res)

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

@st.cache_resource
def get_model():
    model_path = 'model.joblib'
    if SKLEARN:
        if os.path.exists(model_path):
            try:
                return load(model_path)
            except Exception:
                pass
        X, y = generate_synthetic()
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        model.fit(X, y)
        try:
            dump(model, model_path)
        except Exception:
            pass
        return model
    else:
        return HeuristicModel()

model = get_model()

st.title('Winter AI — Snowiness Predictor')
st.write('Adjust the sliders and press Predict to see the model output.')

col1, col2 = st.columns(2)
with col1:
    temp = st.slider('Temperature (°C)', -30.0, 10.0, -5.0, 0.1)
with col2:
    hum = st.slider('Humidity (%)', 0.0, 100.0, 80.0, 1.0)

if st.button('Predict'):
    Xq = np.array([[temp, hum]])
    pred = int(model.predict(Xq)[0])
    probs = model.predict_proba(Xq)[0]
    st.subheader(LABELS.get(pred, 'Unknown'))
    st.write('Probabilities:')
    st.write({LABELS[i]: float(probs[i]) for i in range(len(probs))})

    # small HTML snow visual
    html = f"""
    <div style='position:relative;height:220px;background:linear-gradient(#072033,#071022);overflow:hidden;border-radius:8px;padding:10px'>
    <div style='color:#e6f1ff;font-family:sans-serif'>Prediction: <strong>{LABELS.get(pred)}</strong></div>
    <div id='scene'></div>
    <script>
    // create N snowflakes using for loop
    const scene = document.getElementById('scene');
    const N = {min(80, 20 + int((30 - temp) * 2))};
    for (let i=0;i<N;i++) {{
      const s = document.createElement('div');
      s.textContent = '❄';
      s.style.position='absolute';
      s.style.left = Math.random()*100 + '%';
      s.style.top = Math.random()*-50 + 'px';
      s.style.fontSize = (12 + Math.random()*20) + 'px';
      s.style.opacity = 0.8;
      scene.appendChild(s);
    }}
    // animate with while via requestAnimationFrame
    let last = performance.now();
    function step(now) {
      const dt = (now-last)/1000; last = now;
      let i = 0;
      const flakes = scene.children;
      while (i < flakes.length) {
        const f = flakes[i];
        const y = parseFloat(f.style.top || 0) + 30*dt;
        f.style.top = y + 'px';
        if (y > 220) { scene.removeChild(f); } else { i++; }
      }
      if (flakes.length>0) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
    </script></div>
    """
    st.components.v1.html(html, height=260)
else:
    st.write('Adjust sliders and press Predict to run the model.')
