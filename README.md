# 📦 Amazon Query-Product Ranking System

This project builds a production-ready **query-to-product ranking system** using PyTorch Lightning, SentenceTransformer embeddings, and MLflow for experimentation and deployment.

> 🧪 **Note**: For simplicity, most data preparation, embedding generation, and model hosting are demonstrated directly within **Jupyter notebooks**.

---

## 🚀 Features

- 🧠 **Two-Tower Recommendation Model**
  - Query tower uses sentence embeddings
  - Product tower uses precomputed embeddings

- 🗂️ **Triplet Sampling with Hard Negatives** via Annoy
- 🎯 **Training Pipeline** with metric tracking (NDCG@k, Precision@k, Recall@k)
- 🧪 **MLflow Experiment Tracking**
  - Hyperparameter tuning for `learning_rate`, `num_negative_samples`
  - Artifact versioning

- 🔁 **Model Registry** with auto-promotion logic
- 🌐 **Flask REST API** for online prediction (single + batch)
- 🐳 **Docker-ready deployment**

---

## 🏗️ Project Structure

```text
├── data/                        # Raw data and embeddings
├── outputs/                     # Preprocessed outputs and embeddings
│   ├── product_embeddings/
│   ├── df_products.csv
│   └── encoders/
├── model/                      # Training + evaluation modules
│   ├── model.py
│   └── utils.py
├── server/                     # Flask API for prediction
│   └── app.py
├── notebooks/                  # Jupyter-based dev workflows
├── main.py                     # Training entry point
├── Dockerfile
└── README.md
```

---

## 🧪 Training the Model

```bash
python main.py
```
This will:
- Load triplets and embeddings
- Run training/validation/test
- Track everything in MLflow

---

## 🔍 Query Inference via Flask

```bash
python server/app.py
```

POST endpoint:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless earbuds"}'
```

---

## 🧪 Local Debugging

Use Jupyter for inspecting embeddings:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-uncased')
model.encode("wireless earbuds")
```

---

## ⚙️ Hyperparameter Tuning

MLflow logs runs with:
- `learning_rate`
- `num_negative_samples`

View metrics in:

```bash
mlflow ui --port 5001
```

---

## 📦 Deployment

Build & serve with Docker:

```bash
docker build -t query-ranker .
docker run -p 5000:5000 query-ranker
```

---

## ✅ To-Do

- [ ] Support attention-based fusion
- [ ] Add product metadata reranker
- [ ] Integrate real-time feedback loop
- [ ] Refactor pipeline for proper MLOps setup (config-driven runs, monitoring, CI/CD)

---

## 📚 References
- Amazon ESCI Dataset (Kaggle)
- Sentence-Transformers
- PyTorch Lightning
- MLflow

---

## 👨‍🔬 Author
[Your Name]  
[Your GitHub/Email]

