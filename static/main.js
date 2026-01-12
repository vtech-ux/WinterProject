// main.js — uses for, while, and switch

const predictBtn = document.getElementById('predictBtn');
const resultEl = document.getElementById('result');
const visuals = document.getElementById('visuals');

predictBtn.addEventListener('click', async () => {
  const temp = parseFloat(document.getElementById('temp').value);
  const hum = parseFloat(document.getElementById('hum').value);

  resultEl.textContent = 'Predicting...';
  console.log('Requesting prediction', { temperature: temp, humidity: hum });
  let data;
  try {
    const res = await fetch(`/api/predict?temperature=${encodeURIComponent(temp)}&humidity=${encodeURIComponent(hum)}`);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`API error ${res.status}: ${txt}`);
    }
    data = await res.json();
    // main.js — client-side ML with TensorFlow.js + creative visuals

    const trainBtn = document.getElementById('trainBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultEl = document.getElementById('result');
    const metricsEl = document.getElementById('metrics');
    const visuals = document.getElementById('visuals');
    const tempSlider = document.getElementById('temp');
    const humSlider = document.getElementById('hum');
    const tempVal = document.getElementById('tempVal');
    const humVal = document.getElementById('humVal');
    const modelType = document.getElementById('modelType');

    tempSlider.addEventListener('input', ()=> tempVal.textContent = tempSlider.value);
    humSlider.addEventListener('input', ()=> humVal.textContent = humSlider.value);

    // Synthetic data generator (same logic as earlier but now in JS)
    function generateSynthetic(n=2000, seed=1){
      const X = [];
      const y = [];
      let rnd = seed;
      function rand(){ rnd = (rnd * 9301 + 49297) % 233280; return rnd / 233280; }
      for (let i=0;i<n;i++){
        const t = -20 + rand()*30; // -20..10
        const h = 10 + rand()*90;  // 10..100
        X.push([t,h]);
        if (t <= 0 && h > 70) y.push(2);
        else if (t <= 2 && h > 50) y.push(1);
        else y.push(0);
      }
      return {X,y};
    }

    // Heuristic fallback model (fast)
    class HeuristicModel{
      predict(Xq){
        return Xq.map(([t,h])=> {
          if (t <= 0 && h > 70) return 2;
          if (t <= 2 && h > 50) return 1;
          return 0;
        });
      }
      predictProba(Xq){
        return Xq.map(([t,h])=> {
          if (t <= 0 && h > 70) return [0.05,0.2,0.75];
          if (t <= 2 && h > 50) return [0.1,0.7,0.2];
          return [0.8,0.15,0.05];
        });
      }
    }

    // TensorFlow.js model builder
    async function buildAndTrainTF(X,y){
      // Convert to tensors
      const xs = tf.tensor2d(X);
      const ys = tf.oneHot(tf.tensor1d(y,'int32'),3);

      const model = tf.sequential();
      model.add(tf.layers.dense({units:16,activation:'relu',inputShape:[2]}));
      model.add(tf.layers.dense({units:12,activation:'relu'}));
      model.add(tf.layers.dense({units:3,activation:'softmax'}));
      model.compile({optimizer:tf.train.adam(0.01),loss:'categoricalCrossentropy',metrics:['accuracy']});

      const info = await model.fit(xs,ys,{epochs:30,batchSize:64,shuffle:true,callbacks:{onEpochEnd: (e,l)=>{ if(e%10===0) metricsEl.textContent = `Epoch ${e}: loss=${l.loss.toFixed(3)} acc=${(l.acc||l.accuracy||0).toFixed(3)}` } }});
      xs.dispose(); ys.dispose();
      return model;
    }

    // App state
    let tfModel = null;
    let heuristic = new HeuristicModel();

    trainBtn.addEventListener('click', async ()=>{
      resultEl.textContent = 'Generating data and training...';
      const data = generateSynthetic(2500, Math.floor(Math.random()*10000));
      if (modelType.value === 'tf'){
        tfModel = await buildAndTrainTF(data.X, data.y);
        resultEl.textContent = 'TensorFlow.js model trained in-browser.';
      } else {
        tfModel = null;
        resultEl.textContent = 'Heuristic model selected (no training needed).';
      }
    });

    predictBtn.addEventListener('click', async ()=>{
      const t = parseFloat(tempSlider.value);
      const h = parseFloat(humSlider.value);
      resultEl.textContent = 'Predicting...';

      // choose model
      let pred, probs;
      if (modelType.value === 'heuristic' || !tfModel){
        // demonstrate switch
        const label = (function(){
          const p = heuristic.predict([[t,h]])[0];
          switch(p){
            case 0: return 'No Snow';
            case 1: return 'Light Snow';
            case 2: return 'Heavy Snow';
            default: return 'Unknown';
          }
        })();
        pred = heuristic.predict([[t,h]])[0];
        probs = heuristic.predictProba([[t,h]])[0];
        resultEl.textContent = `${label} (heuristic)`;
      } else {
        // tfModel exists
        const xs = tf.tensor2d([[t,h]]);
        const out = tfModel.predict(xs);
        const arr = await out.data();
        xs.dispose(); out.dispose();
        probs = Array.from(arr);
        // find argmax
        let maxI = 0; for (let i=1;i<probs.length;i++) if (probs[i]>probs[maxI]) maxI=i;
        pred = maxI;
        const labels = ['No Snow','Light Snow','Heavy Snow'];
        resultEl.textContent = `${labels[pred]} (tfjs) — ${probs.map(p=>p.toFixed(2)).join(', ')}`;
      }

      // visuals: create N snowflakes using a for loop that depends on prediction
      visuals.innerHTML = '';
      let N = 20;
      if (pred === 2) N = 120;
      else if (pred === 1) N = 60;

      // use a for loop to create flakes
      for (let i=0;i<N;i++){
        const f = document.createElement('div');
        f.className = 'flake';
        f.textContent = '❄';
        f.style.left = Math.random()*100 + '%';
        f.style.top = Math.random()*-120 + 'px';
        const size = 8 + Math.random()*26;
        f.style.fontSize = size + 'px';
        f.dataset.speed = (0.4 + Math.random()*1.8).toString();
        visuals.appendChild(f);
      }

      // animate with a while loop
      let last = performance.now();
      function step(now){
        const dt = (now-last)/1000; last = now;
        let i=0;
        const flakes = visuals.children;
        while (i < flakes.length){
          const el = flakes[i];
          let top = parseFloat(el.style.top);
          top += parseFloat(el.dataset.speed) * 60 * dt;
          el.style.top = top + 'px';
          el.style.transform = `rotate(${top%360}deg)`;
          if (top > visuals.clientHeight + 40){ visuals.removeChild(el); }
          else i++;
        }
        if (visuals.children.length > 0) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });

    // Small demo function showing while + for usage in a utility: create a winter palette
    function makePalette(){
      const palette = [];
      let i = 0;
      while (i < 6){
        const hue = 190 + i*6;
        palette.push(`hsl(${hue},70%,${30 + i*5}%)`);
        i++;
      }
      for (let j=0;j<palette.length;j++) console.log('palette', j, palette[j]);
      return palette;
    }
    makePalette();
