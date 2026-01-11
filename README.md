Winter AI Demo

A small web app demonstrating a lightweight machine learning model for predicting "snowiness" from temperature and humidity, plus a winter-themed frontend that uses `for`, `while`, and `switch` in JavaScript.

Requirements

- Python 3.8+

Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/ in your browser.

Streamlit version

To run the Streamlit app locally (if `streamlit` is installed):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r project\requirements.txt
streamlit run project/streamlit_app.py
```

To deploy to Streamlit Cloud, push this repo to GitHub and connect it at https://share.streamlit.io
