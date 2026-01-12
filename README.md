Winter AI — Frontend-only

This project is an all-client web app (HTML/CSS/JavaScript) that trains a tiny model in the browser using TensorFlow.js to predict "snowiness" from temperature and humidity. It includes a rich winter visualizer and deliberate uses of `for`, `while`, and `switch` in the JavaScript.

No Python required — open the static files in a browser or serve them with any static server.

Run locally

Open `project/static/index.html` in your browser, or serve the `project/` folder using a simple static server. For example (Node.js must be installed):

```bash
# using a quick npm package
npx serve project

# or with Python's simple HTTP server (works even without Python installed as backend here)
cd project/static
python -m http.server 8000
# then open http://localhost:8000
```

Deployment

- Host as static site on GitHub Pages, Netlify, Vercel, or any static host.
- For Streamlit Cloud or Python hosting, convert accordingly (not needed for this frontend-only demo).

What I changed

- Removed Python/Streamlit backend files.
- Added an enhanced client-side ML demo using TensorFlow.js in `static/`.

Want me to push these frontend-only changes to your GitHub repo `vtech-ux/WinterProject`? If yes, run `gh auth login` then `git push`, or tell me to push and provide auth via `gh` (recommended). 

Streamlit version

To run the Streamlit app locally (if `streamlit` is installed):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r project\requirements.txt
streamlit run project/streamlit_app.py
```

To deploy to Streamlit Cloud, push this repo to GitHub and connect it at https://share.streamlit.io
