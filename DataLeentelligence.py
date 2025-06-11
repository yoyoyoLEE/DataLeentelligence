"""
Data Explorer with LLM Integration
Version: 1.0
Author: Dr. Yong Ha Lee
Description: Interactive data analysis tool with LLM integration for automated insights and abstract generation
Last Updated: June 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from io import BytesIO
from datetime import datetime
import os

# Predefined user credentials
VALID_USERS = {
    "admin": "yonghalee",
    "colecisti": "difficile",
    "psm": "bariatrica",
    "marta": "bonaldi",
    "giovanni": "cesana"
}

# Login function
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

# Check login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# Language settings
if 'language' not in st.session_state:
    st.session_state.language = 'english'

LANGUAGES = {
    'english': {
        'title': "Data Explorer with LLM",
        'upload': "Upload a CSV or Excel file",
        'success': "File uploaded successfully!",
        'stats': "Descriptive statistics",
        'abstract': "Generate Automatic Scientific Abstract",
        'clean': "Clean data (remove nulls, duplicates, basic normalization)",
        'edit': "Direct DataFrame editing",
        'advanced': "Advanced data operations",
        'question': "Ask a question about the data",
        'send': "Send request to LLM",
        'response': "LLM Response",
        'history': "LLM Response History",
        'clear': "Clear history",
        'export': "Export history CSV",
        'visuals': "View exploratory charts",
        'download': "Download modified file",
        'rows_to_analyze': "Number of rows to analyze"
    },
    'italian': {
        'title': "Data Explorer con LLM",
        'upload': "Carica un file CSV o Excel",
        'success': "File caricato con successo!",
        'stats': "Statistiche descrittive",
        'abstract': "Genera Abstract Scientifico Automatico",
        'clean': "Esegui pulizia dei dati (rimozione nulli, duplicati, normalizzazione base)",
        'edit': "Modifica diretta del DataFrame",
        'advanced': "Operazioni avanzate sui dati",
        'question': "Fai a domanda sui dati",
        'send': "Invia richiesta al LLM",
        'response': "Risposta dell'LLM",
        'history': "Cronologia delle Risposte dell'LLM",
        'clear': "Cancella cronologia",
        'export': "Esporta cronologia CSV",
        'visuals': "Visualizza grafici esplorativi",
        'download': "Scarica file modificato",
        'rows_to_analyze': "Numero di righe da analizzare"
    }
}

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Data Explorer with LLM")

# Language selector
col1, col2 = st.columns([4,1])
with col1:
    st.title(LANGUAGES[st.session_state.language]['title'])
with col2:
    lang = st.selectbox("Language", options=['english', 'italian'], index=0 if st.session_state.language == 'english' else 1)
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.rerun()

SAVE_DIR = "./modifiche_auto_salvate"
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize user-specific chat history if not exists
if "user_chat_histories" not in st.session_state:
    st.session_state.user_chat_histories = {}

# Get or create chat history for current user
username = st.session_state.get('username', 'default')
if username not in st.session_state.user_chat_histories:
    st.session_state.user_chat_histories[username] = []

if "trigger_llm" not in st.session_state:
    st.session_state.trigger_llm = False

uploaded_file = st.file_uploader(LANGUAGES[st.session_state.language]['upload'], type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        # Handle data types more carefully
        for col in df.columns:
            # Try to convert to numeric first
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
            
            # Convert remaining non-numeric columns to string
            if not np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].astype(str)
                
        df = df.fillna('')
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        st.stop()

    st.success(LANGUAGES[st.session_state.language]['success'])
    # Show file preview
    st.dataframe(df.head())
    
    st.subheader(LANGUAGES[st.session_state.language]['stats'])
    # Convert all columns to string for safe display
    display_df = df.astype(str)
    st.dataframe(display_df.describe(include='all'))

    rows_to_analyze = st.slider(
        LANGUAGES[st.session_state.language]['rows_to_analyze'],
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

    if st.button(LANGUAGES[st.session_state.language]['abstract']):
        if st.session_state.language == 'english':
            abstract_prompt = f"""Please generate a complete scientific abstract structured with: Introduction, Methods, Results and Conclusion, based on:
1. This dataset (first {rows_to_analyze} rows):
{df.head(rows_to_analyze).to_csv(index=False)}

2. The user's question: {user_question if 'user_question' in locals() else 'No specific question provided'}

3. The AI's analysis: {st.session_state.latest_answer if 'latest_answer' in st.session_state else 'No previous analysis available'}

Focus on synthesizing these three elements into a coherent abstract."""
        else:
            abstract_prompt = f"""Genera un abstract scientifico completo strutturato in: Introduzione, Metodi, Risultati e Conclusione, basato su:
1. Questo dataset (prime {rows_to_analyze} righe):
{df.head(rows_to_analyze).to_csv(index=False)}

2. La domanda dell'utente: {user_question if 'user_question' in locals() else 'Nessuna domanda specifica'}

3. L'analisi precedente dell'AI: {st.session_state.latest_answer if 'latest_answer' in st.session_state else 'Nessuna analisi precedente'}

Concentrati sulla sintesi di questi tre elementi in un abstract coerente."""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "messages": [
                {"role": "system", "content": "You are a scientific assistant expert in writing structured abstracts and articles for academic journals (IMRaD format). Write in proper English with clear structure: Introduction, Methods, Results and Discussion. Use precise scientific language." if st.session_state.language == 'english' else 
                "Sei un assistente scientifico esperte nella redazione di abstract e articoli strutturati per riviste accademiche (formato IMRaD). Scrivi in italiano corretto e formale, con struttura chiara: Introduzione, Metodi, Risultati e Discussione. Evita traduzioni letterali dall'inglese e mantieni un linguaggio scientifico preciso."},
                {"role": "user", "content": abstract_prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.7
        }
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            abstract = result["choices"][0]["message"]["content"]
            st.text_area("Structured Scientific Abstract" if st.session_state.language == 'english' else "Abstract Scientifico Strutturato", 
                        value=abstract, height=400)
            st.download_button("Download Abstract (TXT)" if st.session_state.language == 'english' else "Scarica Abstract (TXT)", 
                             abstract.encode("utf-8"), 
                             file_name="scientific_abstract.txt" if st.session_state.language == 'english' else "abstract_scientifico.txt", 
                             mime="text/plain")
        except Exception as e:
            st.error(f"Errore nella generazione dell'abstract: {e}")
    if st.checkbox(LANGUAGES[st.session_state.language]['clean']):
        df = df.drop_duplicates()
        df = df.dropna(axis=0, how='all')
        # Ensure all columns are treated as strings
        df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x.astype(str).str.strip())
        st.success("Pulizia completata")
        st.dataframe(df.head())

    if st.checkbox(LANGUAGES[st.session_state.language]['edit']):
        st.markdown("Manual editing (use Excel for large volumes)" if st.session_state.language == 'english' else "Modifica manuale (usa Excel per grandi volumi)")
        edited_df = st.experimental_data_editor(df, num_rows="dynamic")
        df = edited_df
        st.success("Modifiche applicate")
        st.dataframe(df.head())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_path = os.path.join(SAVE_DIR, f"autosave_{timestamp}.csv")
        df.to_csv(auto_path, index=False)
        st.info(f"Modifiche salvate automaticamente in: {auto_path}")

    if st.checkbox(LANGUAGES[st.session_state.language]['advanced']):
        if st.button("Group by column" if st.session_state.language == 'english' else "Raggruppa per colonna"):
            group_col = st.selectbox("Select column to group by" if st.session_state.language == 'english' else "Seleziona colonna per raggruppamento", df.columns)
            st.dataframe(df.groupby(group_col).size().reset_index(name='Count' if st.session_state.language == 'english' else 'Conteggio'))
        if st.button("Filter rows" if st.session_state.language == 'english' else "Filtra righe"):
            filter_col = st.selectbox("Select column to filter" if st.session_state.language == 'english' else "Seleziona colonna per filtrare", df.columns)
            filter_val = st.text_input("Value to search" if st.session_state.language == 'english' else "Valore da cercare")
            if filter_val:
                st.dataframe(df[df[filter_col].astype(str).str.contains(filter_val, case=False)])
        if st.button("Merge two columns" if st.session_state.language == 'english' else "Unisci due colonne"):
            col1 = st.selectbox("Prima colonna", df.columns, key='col1')
            col2 = st.selectbox("Seconda colonna", df.columns, key='col2')
            sep = st.text_input("Separatore", value="_")
            new_col_name = st.text_input("Nome nuova colonna", value=f"{col1}_{col2}")
            if col1 and col2 and new_col_name:
                df[new_col_name] = df[col1].astype(str) + sep + df[col2].astype(str)
                st.success(f"Creata nuova colonna: {new_col_name}")
                st.dataframe(df.head())

    user_question = st.text_area(LANGUAGES[st.session_state.language]['question'])

    if st.button(LANGUAGES[st.session_state.language]['send']) and user_question:
        context = f"Ecco i primi {rows_to_analyze} record del dataset:\n{df.head(rows_to_analyze).to_csv(index=False)}\n\nDomanda: {user_question}"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # Enhanced system prompt for statistical analysis
        system_prompt = """You are an AI assistant expert in comprehensive data analysis and scientific writing. 
        When statistical analysis is requested:
        1. Identify and perform all appropriate statistical tests including:
           - Descriptive statistics
           - T-tests (independent, paired)
           - ANOVA and MANOVA
           - Correlation and regression analysis
           - Non-parametric tests
           - Time series analysis
           - Multivariate analysis
        2. Explain the results and their interpretation
        3. Include assumptions checking and post-hoc tests when appropriate
        
        IMPORTANT: 
        - Do NOT include any Python code in your response
        - Only provide the analysis results and explanations
        - Use clear, non-technical language suitable for non-programmers
        
        Always respond in proper English with clear structure. Use formal but understandable language.""" if st.session_state.language == 'english' else """Sei un assistente AI esperto in analisi dati completa e redazione scientifica.
        Quando viene richiesta un'analisi statistica:
        1. Identifica ed esegui tutti i test statistici appropriati inclusi:
           - Statistiche descrittive
           - Test T (indipendenti, appaiati)
           - ANOVA e MANOVA
           - Analisi di correlazione e regressione
           - Test non parametrici
           - Analisi di serie temporali
           - Analisi multivariata
        2. Spiega i risultati e la loro interpretazione
        3. Includi verifica delle assunzioni e test post-hoc quando appropriato
        
        IMPORTANTE:
        - NON includere codice Python nella risposta
        - Fornisci solo i risultati dell'analisi e le spiegazioni
        - Usa un linguaggio chiaro e non tecnico adatto a non programmatori
        
        Scrivi sempre in italiano grammaticalmente perfetto, con struttura chiara."""
        
        payload = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "max_tokens": 3000,
            "temperature": 0.7  # More focused responses
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            st.session_state.user_chat_histories[username].append((user_question, answer))
            st.session_state.latest_answer = answer
        except Exception as e:
            st.session_state.latest_answer = f"Errore durante la chiamata all'API: {e}"

        if "latest_answer" in st.session_state:
            st.markdown(f"### {LANGUAGES[st.session_state.language]['response']}")
            # Remove all code blocks and technical implementation details
            import re
            cleaned_response = re.sub(r'```.*?```', '', st.session_state.latest_answer, flags=re.DOTALL)  # Remove all code blocks
            cleaned_response = re.sub(r'(Here(.*?)code(.*?):|### (Python|Codice) (code|Python|per l\'analisi))', '', cleaned_response, flags=re.IGNORECASE)  # Remove code explanations and headers
            cleaned_response = re.sub(r'(python|codice)(.*?)\n', '', cleaned_response, flags=re.IGNORECASE)  # Remove language mentions
            cleaned_response = re.sub(r'#+\s*(Analysis|Analisi)\s*code\s*#+', '', cleaned_response, flags=re.IGNORECASE)  # Remove analysis code headers
            cleaned_response = cleaned_response.strip()
            st.text_area(LANGUAGES[st.session_state.language]['response'], value=cleaned_response, height=200)

        col1, col2 = st.columns(2)
        with col1:
            if st.download_button("Esporta come TXT", st.session_state.latest_answer.encode("utf-8"), file_name="risposta_ai.txt", mime="text/plain"):
                pass
        with col2:
            from docx import Document
            doc = Document()
            doc.add_heading("Risposta AI", level=1)
            doc.add_paragraph(st.session_state.latest_answer)
            doc_buffer = BytesIO()
            doc.save(doc_buffer)
            doc_buffer.seek(0)
            st.download_button("Esporta come DOCX", doc_buffer, file_name="risposta_ai.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        if st.session_state.user_chat_histories[username]:
            st.markdown(f"### {LANGUAGES[st.session_state.language]['history']}")
            for i, (q, a) in enumerate(reversed(st.session_state.user_chat_histories[username])):
                st.markdown(f"**Q{i+1}:** {q}")
                st.text_area(f"Risposta {i+1}", value=a, height=200, key=f"ans_{i}")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(LANGUAGES[st.session_state.language]['clear']):
                    st.session_state.user_chat_histories[username] = []
                    st.session_state.latest_answer = ""
                    st.rerun()
            with col2:
                if st.button(LANGUAGES[st.session_state.language]['export']):
                    buffer = BytesIO()
                    pd.DataFrame(st.session_state.user_chat_histories[username], columns=["Domanda", "Risposta"]).to_csv(buffer, index=False)
                    st.download_button("Download CSV", buffer.getvalue(), file_name="cronologia_chat.csv", mime="text/csv")

        if st.checkbox(LANGUAGES[st.session_state.language]['visuals']):
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                col1 = st.selectbox("Scegli una variabile numerica per l'istogramma", numeric_cols)
                fig, ax = plt.subplots()
                sns.histplot(df[col1], bins=30, kde=True, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Non ci sono colonne numeriche per il grafico")

        if st.button(LANGUAGES[st.session_state.language]['download']):
            buffer = BytesIO()
            if uploaded_file.name.endswith(".csv"):
                df.to_csv(buffer, index=False)
                st.download_button("Download CSV", buffer.getvalue(), file_name="dati_modificati.csv", mime="text/csv")
            else:
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                buffer.seek(0)
                st.download_button("Download Excel", buffer.getvalue(), file_name="dati_modificati.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: right; font-size: 0.8em; color: #666;">'
        'Made by Dr. Y. Lee'
        '</div>', 
        unsafe_allow_html=True
    )
