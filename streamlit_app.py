import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Winter AI (Static Frontend)', layout='wide')

root = Path(__file__).parent / 'static'
index_path = root / 'index.html'
css_path = root / 'styles.css'
js_path = root / 'main.js'

def inline_frontend():
    html = index_path.read_text(encoding='utf8')
    css = css_path.read_text(encoding='utf8') if css_path.exists() else ''
    js = js_path.read_text(encoding='utf8') if js_path.exists() else ''

    # Inline CSS: replace link tag if present
    html = html.replace('<link rel="stylesheet" href="styles.css" />', f'<style>{css}</style>')

    # Inline main.js by replacing script reference
    html = html.replace('<script src="main.js"></script>', f'<script>{js}</script>')

    # Ensure TensorFlow.js CDN is present; index.html already uses CDN tag so keep as-is
    return html


st.title('Winter AI — Static Frontend (served in Streamlit)')
st.write('This page embeds the client-side app (HTML/CSS/JS) inside Streamlit using an HTML component.')

html = inline_frontend()

st.components.v1.html(html, height=920, scrolling=True)
