# Clasificador de Objetivos de Desarrollo Sostenible (ODS)

Aplicación web que clasifica textos en español según el Objetivo de Desarrollo Sostenible (ODS) con el que guardan mayor relación semántica.

## Cómo funciona

1. El usuario ingresa un texto libre en español.
2. El modelo procesa el texto mediante un pipeline de NLP (TF-IDF + TruncatedSVD + LinearSVC).
3. La app devuelve los 3 ODS más probables con su nivel de confianza.

## Pipeline del modelo

| Paso | Componente | Descripción |
|------|-----------|-------------|
| 1 | `TfidfVectorizer` | Vectorización con unigramas y bigramas, preprocesamiento integrado |
| 2 | `TruncatedSVD` | Reducción a 300 componentes |
| 3 | `LinearSVC` | Clasificador lineal (C=0.5, class_weight='balanced') |

**Rendimiento en el conjunto de prueba (1,932 textos):**

- F1 macro: **0.846**
- Accuracy: **87.6 %**
- Acierto en top-3: **97.6 %**

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Autores

- Sara Rivera
- Juan Pablo Cuesta Vanegas

Proyecto del curso de Aprendizaje No Supervisado — Maestría en Inteligencia Artificial, Universidad de los Andes.
