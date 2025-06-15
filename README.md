# Page Relevance - Análise de Sentimentos para Turismo

Este projeto realiza análise de sentimentos e relevância de páginas de turismo, utilizando modelos de linguagem (transformers) e técnicas de NLP para classificar conteúdos como positivos, neutros ou negativos.

## Como usar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure as fontes:**
   - Edite o arquivo `config/source.json` com as URLs dos sites a serem analisados.
3. **Execute a análise:**
   ```bash
   python src/run.py
   ```
4. **Veja os resultados:**
   - Os resultados estarão na pasta `results/` (adicionada ao `.gitignore`).

## Estrutura
```
page_relevance/
├── src/
│   ├── analyzer.py
│   ├── analyzer_bart.py
│   ├── scraper.py
│   └── ...
├── config/
│   └── source.json
├── results/         # (ignorado no git)
├── requirements.txt
├── README.md
├── .gitignore
└── ...
```

## Observações
- Não suba arquivos grandes ou dados sensíveis.
- O projeto suporta GPU (CUDA) para acelerar a análise.
- Para dúvidas, consulte os comentários no código ou abra uma issue.
