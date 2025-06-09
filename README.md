# 🧪 Data Explorer with LLM Integration

**Version:** 1.0  
**Author:** Dr. Yongha Lee  
**Last Updated:** June 2025

An interactive, multilingual data analysis webapp powered by **Streamlit** and integrated with **LLM APIs**. Designed for healthcare professionals, data scientists, and researchers, this app allows users to upload tabular datasets (CSV/Excel), analyze them visually and statistically, ask questions using natural language, and even generate scientific abstracts — all in a clean, browser-based interface.

---

## 🚀 Features

- 🔐 Secure login system (basic credential-based)
- 📄 Upload CSV or Excel files
- 📊 Descriptive statistics and editable data previews
- 💬 LLM-powered Q&A interface
- 🧠 Automatic scientific abstract generation (IMRaD format)
- 🧹 Basic data cleaning (null/duplicate removal, normalization)
- ✏️ Inline data editor (via Streamlit’s interactive editor)
- 📁 Export cleaned data, AI responses, and chat history (CSV, TXT, DOCX)
- 📈 Visualizations with `matplotlib` and `seaborn`
- 🌍 Multilingual UI: English and Italian

---

## 🧰 Tech Stack

- Python 3.11+
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [OpenRouter API](https://openrouter.ai/) (for LLM integration)

---

## 🛠️ Installation (Run Locally)

---

## 🔐 Access Control

The app includes a **basic login system** with a limited set of predefined users for demonstration purposes.

> 🚨 For production use, it is strongly recommended to implement a secure authentication mechanism (e.g., OAuth, hashed credentials).


---

## 🙌 Acknowledgements

This app uses models and API routing via [OpenRouter.ai](https://openrouter.ai/), enabling integration with models such as DeepSeek and Qwen.

---

**Made with ❤️ by Dr. Yongha Lee**
