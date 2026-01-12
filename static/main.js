// Advanced frontend ML: more features, larger data, training charts, persistence

// DOM refs
const trainBtn = document.getElementById('trainBtn');
const stopBtn = document.getElementById('stopBtn');
const saveBtn = document.getElementById('saveBtn');
const downloadBtn = document.getElementById('downloadBtn');
const loadBtn = document.getElementById('loadBtn');
const resultEl = document.getElementById('result');
const metricsEl = document.getElementById('metrics');
const visuals = document.getElementById('visuals');
const tempSlider = document.getElementById('temp');
const humSlider = document.getElementById('hum');
const tempVal = document.getElementById('tempVal');
const humVal = document.getElementById('humVal');
const modelType = document.getElementById('modelType');
const trainSizeEl = document.getElementById('trainSize');
const trainSizeVal = document.getElementById('trainSizeVal');
const epochsEl = document.getElementById('epochs');
const batchSizeEl = document.getElementById('batchSize');

// chart containers
const lossChart = document.getElementById('lossChart');
const accChart = document.getElementById('accChart');

tempSlider.addEventListener('input', ()=> tempVal.textContent = tempSlider.value);
humSlider.addEventListener('input', ()=> humVal.textContent = humSlider.value);
trainSizeEl.addEventListener('input', ()=> trainSizeVal.textContent = trainSizeEl.value);

let trainingController = {stop:false};

// Enhanced synthetic dataset with extra features
function generateSyntheticEnhanced(n=2500, seed=1){
  const X = [];
  const y = [];
  let rnd = seed;
  function rand(){ rnd = (rnd * 9301 + 49297) % 233280; return rnd / 233280; }
  for (let i=0;i<n;i++){
    const temp = -30 + rand()*40; // -30..10
    const hum = 5 + rand()*95;    // 5..100
    const pressure = 980 + rand()*70; // hPa
    const wind = rand()*20; // m/s
    const elev = -50 + rand()*4000; // meters
    // derived features
    const dewPoint = temp - ((100 - hum)/5);
    const feelsLike = temp - wind*0.1;
    const seasonal = Math.sin(i/50) * 5;
    const features = [temp, hum, pressure, wind, elev, dewPoint, feelsLike, seasonal];
    X.push(features);
    // label rules (stronger combining multiple features)
    if (temp <= -2 && hum > 68 && wind < 12) y.push(2);
    else if (temp <= 1 && hum > 55) y.push(1);
    else y.push(0);
  }
  return {X,y};
}

// normalization utilities
function computeStats(X){
  const dims = X[0].length;
  const mean = new Array(dims).fill(0);
  const std = new Array(dims).fill(0);
  const n = X.length;
  for (let j=0;j<dims;j++){
    for (let i=0;i<n;i++) mean[j] += X[i][j];
    mean[j] /= n;
    for (let i=0;i<n;i++){ const d = X[i][j] - mean[j]; std[j] += d*d; }
    std[j] = Math.sqrt(std[j]/n) || 1.0;
  }
  return {mean,std};
}
function normalize(X, stats){
  return X.map(row => row.map((v,i)=> (v - stats.mean[i]) / stats.std[i]));
}

// Heuristic fallback for quick predictions
class HeuristicModel{
  predict(Xq){
    return Xq.map(row=> {
      const t = row[0], h = row[1];
      if (t <= -2 && h > 68) return 2;
      if (t <= 1 && h > 55) return 1;
      return 0;
    });
  }
  predictProba(Xq){
    return Xq.map(row=> {
      const p = this.predict([row])[0];
      if (p===2) return [0.05,0.2,0.75];
      if (p===1) return [0.1,0.7,0.2];
      return [0.8,0.15,0.05];
    });
  }
}

// build a deeper TF.js model
function buildModel(inputDim){
  const model = tf.sequential();
  model.add(tf.layers.dense({units:64,activation:'relu',inputShape:[inputDim]}));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({rate:0.25}));
  model.add(tf.layers.dense({units:48,activation:'relu'}));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({rate:0.2}));
  model.add(tf.layers.dense({units:32,activation:'relu'}));
  model.add(tf.layers.dense({units:3,activation:'softmax'}));
  model.compile({optimizer:tf.train.adam(0.005),loss:'categoricalCrossentropy',metrics:['accuracy']});
  return model;
}

// helper: convert arrays to tensors
function toTensors(X,y){
  const xs = tf.tensor2d(X);
  const ys = tf.oneHot(tf.tensor1d(y,'int32'),3);
  return {xs,ys};
}

// metrics functions
function confusionMatrix(trueY, predY){
  const K = 3; const M = Array.from({length:K}, ()=> Array(K).fill(0));
  for (let i=0;i<trueY.length;i++){ M[trueY[i]][predY[i]]++; }
  return M;
}
function classificationReport(trueY, predY){
  const K = 3; const precision = [], recall = [], f1 = [];
  for (let c=0;c<K;c++){
    let tp=0, fp=0, fn=0;
    for (let i=0;i<trueY.length;i++){
      if (predY[i]===c && trueY[i]===c) tp++;
      if (predY[i]===c && trueY[i]!==c) fp++;
      if (predY[i]!==c && trueY[i]===c) fn++;
    }
    const p = tp/(tp+fp||1); const r = tp/(tp+fn||1); const f = 2*p*r/(p+r||1);
    precision.push(p); recall.push(r); f1.push(f);
  }
  return {precision,recall,f1};
}

// plotting helpers using Plotly (loaded from CDN in HTML)
function initCharts(){
  Plotly.newPlot(lossChart, [{x:[], y:[], mode:'lines', name:'loss'}], {margin:{t:10,b:30}});
  Plotly.newPlot(accChart, [{x:[], y:[], mode:'lines', name:'accuracy'}], {margin:{t:10,b:30}});
}
function updateCharts(epoch, loss, acc){
  Plotly.extendTraces(lossChart, {x:[[epoch]], y:[[loss]]}, [0]);
  Plotly.extendTraces(accChart, {x:[[epoch]], y:[[acc]]}, [0]);
}

initCharts();

let tfModel = null;
let heuristic = new HeuristicModel();

async function trainModel(){
  const n = parseInt(trainSizeEl.value,10);
  const epochs = parseInt(epochsEl.value,10);
  const batchSize = parseInt(batchSizeEl.value,10);
  trainingController.stop = false;
  trainBtn.disabled = true; stopBtn.disabled = false; saveBtn.disabled = true; downloadBtn.disabled=true;
  resultEl.textContent = 'Preparing synthetic dataset...';

  const data = generateSyntheticEnhanced(n, Math.floor(Math.random()*1e6));
  const split = Math.floor(n*0.8);
  const XtrainRaw = data.X.slice(0,split);
  const ytrain = data.y.slice(0,split);
  const XvalRaw = data.X.slice(split);
  const yval = data.y.slice(split);

  const stats = computeStats(XtrainRaw);
  const Xtrain = normalize(XtrainRaw, stats);
  const Xval = normalize(XvalRaw, stats);

  if (modelType.value === 'heuristic'){
    tfModel = null;
    resultEl.textContent = 'Heuristic selected — ready.';
    trainBtn.disabled = false; stopBtn.disabled = true; saveBtn.disabled = true; downloadBtn.disabled=true;
    return;
  }

  resultEl.textContent = 'Building model...';
  tfModel = buildModel(Xtrain[0].length);

  const {xs,ys} = toTensors(Xtrain,ytrain);
  const {xs: vx, ys: vy} = toTensors(Xval,yval);

  await tfModel.fit(xs, ys, {
    epochs,
    batchSize,
    validationData:[vx,vy],
    shuffle:true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        updateCharts(epoch+1, logs.loss, logs.acc || logs.accuracy || logs.val_accuracy || 0);
        metricsEl.innerText = `Epoch ${epoch+1}: loss=${(logs.loss||0).toFixed(3)} val_loss=${(logs.val_loss||0).toFixed(3)}`;
        if (trainingController.stop) {
          await tfModel.stopTraining?.();
        }
      }
    }
  });

  xs.dispose(); ys.dispose(); vx.dispose(); vy.dispose();

  // evaluate on validation set
  const preds = [];
  for (let i=0;i<Xval.length;i+=1){
    const out = tfModel.predict(tf.tensor2d([Xval[i]]));
    const arr = await out.data(); out.dispose();
    let maxI = 0; for (let j=1;j<arr.length;j++) if (arr[j]>arr[maxI]) maxI=j;
    preds.push(maxI);
  }
  const cm = confusionMatrix(yval, preds);
  const rep = classificationReport(yval, preds);
  resultEl.textContent = `Validation acc ${(preds.filter((p,i)=>p===yval[i]).length / yval.length *100).toFixed(1)}%`;
  metricsEl.innerHTML = `<pre>Precision: ${rep.precision.map(v=>v.toFixed(2)).join(', ')}\nRecall: ${rep.recall.map(v=>v.toFixed(2)).join(', ')}\nF1: ${rep.f1.map(v=>v.toFixed(2)).join(', ')}</pre>`;

  // show confusion matrix simple table
  const cmHtml = ['<table class="card small"><tr><th></th><th>Pred:0</th><th>Pred:1</th><th>Pred:2</th></tr>'];
  for (let i=0;i<cm.length;i++){
    cmHtml.push(`<tr><th>True:${i}</th>${cm[i].map(c=>`<td>${c}</td>`).join('')}</tr>`);
  }
  cmHtml.push('</table>');
  metricsEl.innerHTML += cmHtml.join('');

  trainBtn.disabled = false; stopBtn.disabled = true; saveBtn.disabled = false; downloadBtn.disabled=false;
}

trainBtn.addEventListener('click', ()=> trainModel());
stopBtn.addEventListener('click', ()=> { trainingController.stop = true; stopBtn.disabled = true; });

saveBtn.addEventListener('click', async ()=>{
  if (!tfModel) return alert('No tf model to save');
  await tfModel.save('indexeddb://winter-model');
  alert('Model saved to IndexedDB (winter-model)');
});

downloadBtn.addEventListener('click', async ()=>{
  if (!tfModel) return alert('No tf model to download');
  await tfModel.save('downloads://winter-model');
});

loadBtn.addEventListener('click', async ()=>{
  try{
    const m = await tf.loadLayersModel('indexeddb://winter-model');
    if (m){ tfModel = m; alert('Loaded model from IndexedDB'); saveBtn.disabled=false; downloadBtn.disabled=false; }
  } catch(e){ alert('No model found in IndexedDB'); }
});

// Predict button reacts to current model or heuristic
document.getElementById('temp').addEventListener('input', ()=> document.getElementById('tempVal').textContent = document.getElementById('temp').value);
predictBtn.addEventListener('click', async ()=>{
  const t = parseFloat(tempSlider.value);
  const h = parseFloat(humSlider.value);
  resultEl.textContent = 'Predicting...';

  if (modelType.value === 'heuristic' || !tfModel){
    const p = heuristic.predict([[t,h]])[0];
    const mapping = ['No Snow','Light Snow','Heavy Snow'];
    resultEl.textContent = `${mapping[p]} (heuristic)`;
  } else {
    // need stats to normalize; simplest: regenerate small stats from sample dataset of trainSize
    const statsSample = computeStats(generateSyntheticEnhanced(parseInt(trainSizeEl.value,10),1).X);
    const row = normalize([[t,h, 1013, 2, 100, t-((100-h)/5), t-0.2*2, 0]], statsSample)[0];
    const xs = tf.tensor2d([row]);
    const out = tfModel.predict(xs);
    const arr = await out.data(); xs.dispose(); out.dispose();
    let maxI = 0; for (let i=1;i<arr.length;i++) if (arr[i]>arr[maxI]) maxI=i;
    const mapping = ['No Snow','Light Snow','Heavy Snow'];
    resultEl.textContent = `${mapping[maxI]} (tfjs) — ${arr.map(v=>v.toFixed(2)).join(', ')}`;
  }
  // enhanced visuals based on selection
  const predLabel = resultEl.textContent;
  renderVisuals(predLabel);
});

function renderVisuals(predLabel){
  visuals.innerHTML = '';
  let N = 30;
  if (predLabel.includes('Heavy')) N = 180; else if (predLabel.includes('Light')) N = 80;
  for (let i=0;i<N;i++){
    const f = document.createElement('div'); f.className='flake'; f.textContent='❄';
    f.style.left = Math.random()*100 + '%'; f.style.top = Math.random()*-300 + 'px';
    f.style.fontSize = (8 + Math.random()*28) + 'px'; f.dataset.speed = (0.2 + Math.random()*2).toString();
    visuals.appendChild(f);
  }
  let last = performance.now();
  function step(now){ const dt=(now-last)/1000; last=now; let i=0; const flakes=visuals.children; while(i<flakes.length){ const el=flakes[i]; let top = parseFloat(el.style.top); top += parseFloat(el.dataset.speed)*60*dt; el.style.top = top + 'px'; if (top > visuals.clientHeight+40) visuals.removeChild(el); else i++; } if (visuals.children.length>0) requestAnimationFrame(step); }
  requestAnimationFrame(step);
}

// init small palette
function makePalette(){ const palette=[]; let i=0; while(i<6){ palette.push(`hsl(${190+i*6},70%,${30+i*5}%)`); i++; } return palette; }
makePalette();
