from transformers import pipeline
import re

def analyze_term_sentiments(site_text, terms, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Analisa o sentimento do contexto de cada termo em 'terms' dentro do texto do site.
    Retorna um dicionário {termo: sentimento_médio}.
    """
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
    results = {}

    for term in terms:
        # Encontra frases que contêm o termo (simples split por ponto)
        contexts = [sent for sent in re.split(r'[.!?]', site_text) if term.lower() in sent.lower()]
        if not contexts:
            results[term] = None
            continue
        sentiments = []
        for ctx in contexts:
            try:
                res = sentiment_analyzer(ctx.strip())[0]
                # Converte o label para score numérico
                if "star" in res["label"]:
                    score = int(res["label"][0]) / 5
                elif res["label"].lower() in ["positive", "pos"]:
                    score = 1.0
                elif res["label"].lower() in ["negative", "neg"]:
                    score = 0.0
                else:
                    score = 0.5
                sentiments.append(score)
            except Exception:
                continue
        results[term] = sum(sentiments) / len(sentiments) if sentiments else None
    return results

# Exemplo de uso
if __name__ == "__main__":
    texto = "A praia é linda. O hotel foi ruim. O passeio foi maravilhoso. A praia estava cheia."
    termos = ["praia", "hotel", "passeio"]
    print(analyze_term_sentiments(texto, termos))
