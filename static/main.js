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
    console.log('Prediction response', data);
  } catch (err) {
    console.error('Prediction failed', err);
    resultEl.textContent = 'Prediction failed: ' + err.message;
    return;
  }

  // Use switch to interpret label
  let message = '';
  switch (data.prediction) {
    case 0:
      message = `Result: ${data.label} — It's unlikely to snow.`;
      break;
    case 1:
      message = `Result: ${data.label} — Light snow possible.`;
      break;
    case 2:
      message = `Result: ${data.label} — Heavy snow likely!`;
      break;
    default:
      message = 'Unknown result';
  }

  resultEl.textContent = `${message} (P: ${data.probabilities.map(p=>p.toFixed(2)).join(', ')})`;

  // Create a winter visual — generate N snowflakes using a for loop
  visuals.innerHTML = '';
  const N = Math.min(80, 20 + Math.round((100 - data.input.temperature) / 2));
  for (let i = 0; i < N; i++) {
    const s = document.createElement('div');
    s.className = 'snowflake';
    s.textContent = '❄';
    s.style.left = Math.random() * 100 + '%';
    s.style.top = (Math.random() * -50) + 'px';
    s.dataset.speed = (0.5 + Math.random() * 1.5).toString();
    s.dataset.x = (Math.random() * visuals.clientWidth).toString();
    s.dataset.y = (-Math.random() * 200).toString();
    s.style.transform = `translate(${s.dataset.x}px, ${s.dataset.y}px)`;
    visuals.appendChild(s);
  }

  // Animate using a while loop with requestAnimationFrame
  let running = true;
  let last = performance.now();

  function step(now) {
    const dt = (now - last) / 1000; last = now;
    // While we still have flakes, move them
    let i = 0;
    const flakes = visuals.children;
    while (i < flakes.length) {
      const f = flakes[i];
      let x = parseFloat(f.dataset.x);
      let y = parseFloat(f.dataset.y);
      const speed = parseFloat(f.dataset.speed);
      y += 30 * speed * dt; // fall
      x += Math.sin((now / 1000) + i) * 8 * dt; // drift
      f.dataset.x = x.toString();
      f.dataset.y = y.toString();
      f.style.transform = `translate(${x}px, ${y}px)`;
      // remove when below the box
      if (y > visuals.clientHeight + 40) {
        visuals.removeChild(f);
        // do not increment i because items shifted
      } else {
        i++;
      }
    }

    if (running && visuals.children.length > 0) {
      requestAnimationFrame(step);
    } else {
      running = false;
    }
  }

  if (!running) {
    running = true; last = performance.now(); requestAnimationFrame(step);
  } else {
    last = performance.now(); requestAnimationFrame(step);
  }

});

// Small demo of while loop and for loop combined: generate a small palette
function makePalette() {
  const palette = [];
  let i = 0;
  // while loop to push 5 colors
  while (i < 5) {
    const hue = 190 + i * 8; // wintery blues
    palette.push(`hsl(${hue}, 70%, ${40 + i * 6}%)`);
    i++;
  }
  // for loop to print them (console demo)
  for (let j = 0; j < palette.length; j++) {
    console.log('Palette color', j, palette[j]);
  }
  return palette;
}

makePalette();
