# 🏖️ PROJETO SAVINO - Análise de Relevância para Turismo

Sistema inteligente de análise de conteúdo turístico utilizando **IA/NLP** (BERT, RoBERTa) para classificar relevância de sites sobre destinos como Porto de Galinhas.

## 🎯 Público-Alvo
- **Viajantes lifestyle** 🌟
- **Casais românticos** 💕  
- **Aventureiros aquáticos** 🏄‍♀️
- **Foco custo-benefício** 💰

## 🚀 Como usar

### 1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

### 2. **Configurar fontes:**
- Edite `config/sources.json` com URLs dos sites turísticos

### 3. **Executar análise:**
```bash
# Análise completa + Power BI
python src/run.py

# Análise simples
python src/scrapers/run.py

# Com medição de tempo
python src/tempo.py
```

### 4. **Resultados:**
- **`results/`** - Análises JSON detalhadas
- **`power_bi_data/`** - CSVs para dashboards

## 📂 Estrutura Final

```
page_relevance/
├── src/
│   ├── run.py                    # 🎯 Entry point principal + Power BI
│   ├── init.py                   # 🎛️ Orquestrador central
│   ├── scraper.py               # 🕷️ Web scraping + processamento
│   ├── analyzer.py              # 🧠 IA/NLP (BERT português)
│   ├── utils.py                 # 🔧 Funções auxiliares
│   ├── data_integration.py      # 📊 Geração CSVs Power BI
│   ├── term_sentiment.py       # 💭 Análise sentimento termos
│   ├── tempo.py                 # ⏱️ Entry point com benchmark
│   └── scrapers/
│       └── run.py               # 🏃 Entry point simplificado
├── config/
│   └── sources.json             # 🌐 URLs dos sites
├── power_bi_data/               # 📈 CSVs para dashboards
│   ├── sites.csv
│   ├── terms_analysis.csv
│   ├── site_metrics.csv
│   └── category_metrics.csv
├── results/                     # 📋 Resultados JSON (git ignored)
├── requirements.txt             # 📦 Dependências Python
└── README.md
```

## 🔄 Fluxo de Análise

### **Fase 1-2:** Inicialização + Scraping
- Carregamento de modelos IA (BERT português)
- Extração de conteúdo com BeautifulSoup

### **Fase 3:** Pré-processamento
- Normalização (minúsculas, sem acentos)
- Remoção de stopwords (NLTK)
- Tokenização e limpeza

### **Fase 4:** Análise de Sentimento
- Pipeline Transformers com modelo português
- Processamento em chunks com agregação
- Sistema híbrido: IA + fallback estatístico

### **Fase 5:** Categorização + Relevância
- Zero-shot classification (8 categorias)
- Cálculo: `(frequência ÷ total) × 1000 × multiplicador`
- Score 0-10 por categoria de importância

### **Fase 6:** Consolidação + Relatórios
- 3 tipos de relatórios (JSON detalhado, TXT resumo, insights)
- CSVs automáticos para Power BI
- Backup progressivo e recuperação

## 🧮 Cálculo de Relevância

```python
Score = (Frequência ÷ Total Palavras) × 1000 × Multiplicador

# Multiplicadores por categoria:
destination = 2.0      # Porto de Galinhas
timing = 1.8          # Melhor época
price = 1.7           # Preços, custos
accommodation = 1.6   # Hotéis, pousadas
activity = 1.5        # Mergulho, passeios
```

## 🤖 Tecnologias

- **IA/ML:** Transformers, BERT, RoBERTa, BART
- **NLP:** NLTK, Zero-shot classification
- **Web:** BeautifulSoup, Requests
- **Data:** Pandas, JSON, CSV
- **Viz:** Power BI integration

## ⚡ Performance

- **GPU (CUDA)** - Aceleração automática quando disponível
- **Processamento híbrido** - IA + fallback estatístico
- **Backup progressivo** - Recuperação de falhas automática
- **Chunks inteligentes** - Textos grandes divididos otimamente

## 📊 Outputs

### **Análises JSON:**
- `analysis_detailed_[ID].json` - Dados completos
- `analysis_report_[ID].txt` - Resumo legível
- `analysis_consolidated_[ID].json` - Insights comparativos

### **Power BI CSVs:**
- `terms_analysis.csv` - Termos por site/categoria
- `site_metrics.csv` - Métricas gerais por site
- `category_metrics.csv` - Performance por categoria
- `travel_terms.csv` - Termos turísticos consolidados

## 🛠️ Troubleshooting

- **Erro de modelo:** Instale transformers com `pip install transformers torch`
- **GPU não detectada:** Instale `torch` com suporte CUDA
- **Memória insuficiente:** Use `python src/scrapers/run.py` (versão leve)
- **Sites inacessíveis:** Verifique `config/sources.json`

---

**Desenvolvido para análise inteligente de conteúdo turístico com foco em Porto de Galinhas** 🏖️