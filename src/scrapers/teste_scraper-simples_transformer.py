from collections import Counter
from transformers import pipeline
import torch

class TextAnalyzer:
    def __init__(self):
        # Keywords tradicionais como fallback
        self.destination_keywords = ['porto', 'galinhas', 'pernambuco', 'recife', 'nordeste', 'brasil']
        self.timing_keywords = ['época', 'quando', 'mês', 'temporada', 'período', 'clima']
        self.activity_keywords = ['mergulho', 'praia', 'passeio', 'turismo', 'viagem', 'hotel', 'pousada']
        self.price_keywords = ['preço', 'custo', 'valor', 'barato', 'caro', 'orçamento']
        self.accommodation_keywords = ['hospedagem', 'hotéis', 'pousada', 'resort', 'hostel', 'acomodação']
        
        self.category_multipliers = {
            'destination': 2.0,
            'timing': 1.8,
            'activity': 1.5,
            'price': 1.7,
            'accommodation': 1.6,
            'general': 1.0
        }
        
        # Categorias em português para melhor classificação
        self.contextual_categories = [
            "localização e destino turístico",
            "tempo e sazonalidade de viagem", 
            "atividades e experiências turísticas",
            "preços e custos de viagem",
            "sentimentos e opiniões sobre viagem",
            "planejamento e logística de viagem",
            "hospedagem e acomodação",
            "gastronomia e restaurantes"
        ]
        
        # Inicializar modelos em português
        try:
            # Modelo de sentimento em português
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="neuralmind/bert-base-portuguese-cased",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Modelo de sentimento em português carregado!")
        except Exception as e:
            print(f"❌ Erro no modelo de sentimento: {e}")
            print("🔄 Tentando modelo alternativo...")
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✅ Modelo de sentimento alternativo carregado!")
            except Exception as e2:
                print(f"❌ Erro no modelo alternativo: {e2}")
                self.sentiment_pipeline = None

        try:
            # Modelo de classificação - mantendo BART que funciona bem com português
            self.classifier_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Modelo de classificação carregado!")
        except Exception as e:
            print(f"❌ Erro no modelo de classificação: {e}")
            self.classifier_pipeline = None

    def categorize_keyword_contextual(self, term, context_text=""):
        """Categoriza usando contexto neutro e categorias mais distintas"""
        
        if not self.classifier_pipeline:
            return self.categorize_keyword_fallback(term)
        
        # Contexto mais neutro, sem viés para destino
        analysis_text = f"Esta palavra ou frase '{term}' em contexto de viagem se refere a: {context_text[:500]}"
        
        try:
            result = self.classifier_pipeline(analysis_text, self.contextual_categories)
            
            # Mapear categorias mais distintas para simplificadas
            category_mapping = {
                "destinos e lugares turísticos": "destination",
                "datas e períodos de viagem": "timing", 
                "atividades de lazer e turismo": "activity",
                "valores monetários e preços": "price",
                "opiniões e avaliações": "sentiment",
                "organização de viagens": "planning",
                "hotéis e onde ficar": "accommodation",
                "comida e restaurantes": "food"
            }
            
            best_category = result['labels'][0]
            confidence = result['scores'][0]
            
            # Aumentar threshold para ser mais seletivo
            if confidence > 0.5:  # Voltou para 0.5 para maior precisão
                simple_category = category_mapping.get(best_category, "general")
                return {
                    'category': simple_category,
                    'detailed_category': best_category,
                    'confidence': round(confidence, 3),
                    'method': 'contextual'
                }
            else:
                return self.categorize_keyword_fallback(term)
                
        except Exception as e:
            print(f"Erro na classificação contextual: {e}")
            return self.categorize_keyword_fallback(term)

    def categorize_keyword_fallback(self, term):
        """Método fallback usando palavras-chave expandidas"""
        if any(keyword in term.lower() for keyword in self.destination_keywords):
            category = 'destination'
        elif any(keyword in term.lower() for keyword in self.timing_keywords):
            category = 'timing'
        elif any(keyword in term.lower() for keyword in self.activity_keywords):
            category = 'activity'
        elif any(keyword in term.lower() for keyword in self.price_keywords):
            category = 'price'
        elif any(keyword in term.lower() for keyword in self.accommodation_keywords):
            category = 'accommodation'
        else:
            category = 'general'
            
        return {
            'category': category,
            'detailed_category': category,
            'confidence': 0.7,
            'method': 'keyword'
        }

    def analyze_sentiment_transformer(self, text):
        if not self.sentiment_pipeline:
            return self.calculate_sentiment_fallback(text)
        
        # Processar em chunks menores para BERT
        max_length = 400  # BERT tem limite menor que RoBERTa
        text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        chunk_sentiments = []
        
        for chunk in text_chunks:
            if len(chunk.strip()) < 20:  # Aumentar mínimo para português
                continue
                
            try:
                result = self.sentiment_pipeline(chunk)
                
                # Mapear labels do BERT português
                if 'POSITIVE' in result[0]['label'].upper() or 'LABEL_2' in result[0]['label']:
                    score = result[0]['score']
                elif 'NEGATIVE' in result[0]['label'].upper() or 'LABEL_0' in result[0]['label']:
                    score = 1 - result[0]['score']
                else:  # Neutral
                    score = 0.5
                
                chunk_sentiments.append(score)
                
            except Exception as e:
                print(f"Erro na análise de sentimento chunk: {e}")
                continue
        
        if not chunk_sentiments:
            return self.calculate_sentiment_fallback(text)
        
        overall_sentiment = sum(chunk_sentiments) / len(chunk_sentiments)
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_confidence': round(sum(chunk_sentiments) / len(chunk_sentiments), 3),
            'chunks_analyzed': len(chunk_sentiments),
            'method': 'transformer'
        }

    def calculate_sentiment_fallback(self, text):
        positive_words = ['melhor', 'bom', 'boa', 'excelente', 'incrível', 'lindo', 'maravilhoso', 'ótimo', 'perfeito']
        negative_words = ['ruim', 'péssimo', 'caro', 'problema', 'difícil', 'complicado', 'terrível', 'horrível']
        
        positive_count = sum(text.lower().count(word) for word in positive_words)
        negative_count = sum(text.lower().count(word) for word in negative_words)
        
        if positive_count + negative_count > 0:
            score = positive_count / (positive_count + negative_count)
        else:
            score = 0.5
            
        return {
            'overall_sentiment': score,
            'sentiment_confidence': 0.5,
            'chunks_analyzed': 1,
            'method': 'fallback'
        }

    def calculate_sentiment(self, text):
        sentiment_analysis = self.analyze_sentiment_transformer(text)
        return sentiment_analysis['overall_sentiment']

    def calculate_relevance_score(self, term, frequency, total_words, category):
        base_score = (frequency / total_words) * 1000
        multiplier = self.category_multipliers.get(category, 1.0)
        relevance_score = base_score * multiplier
        return round(min(relevance_score, 10.0), 2)

    def analyze_ngrams(self, filtered_words, total_relevant_words, original_text=""):
        """Análise com contexto expandido para melhor categorização"""
        unigrams = Counter(filtered_words)
        
        bigrams = []
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
            bigrams.append(bigram)
        bigrams_count = Counter(bigrams)
        
        trigrams = []
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
            trigrams.append(trigram)
        trigrams_count = Counter(trigrams)
        
        all_terms = []
        
        # Unigrams com contexto expandido
        for term, freq in unigrams.most_common(30):
            category_info = self.categorize_keyword_contextual(term, original_text)
            relevance_score = self.calculate_relevance_score(term, freq, total_relevant_words, category_info['category'])
            
            all_terms.append({
                'term': term,
                'type': 'unigram',
                'frequency': freq,
                'relevance_score': relevance_score,
                'category': category_info['category'],
                'detailed_category': category_info['detailed_category'],
                'classification_confidence': category_info['confidence'],
                'classification_method': category_info['method']
            })
        
        # Bigrams com contexto expandido
        for term, freq in bigrams_count.most_common(20):
            category_info = self.categorize_keyword_contextual(term, original_text)
            relevance_score = self.calculate_relevance_score(term, freq, total_relevant_words, category_info['category'])
            
            all_terms.append({
                'term': term,
                'type': 'bigram',
                'frequency': freq,
                'relevance_score': relevance_score,
                'category': category_info['category'],
                'detailed_category': category_info['detailed_category'],
                'classification_confidence': category_info['confidence'],
                'classification_method': category_info['method']
            })
        
        # Trigrams com contexto expandido
        for term, freq in trigrams_count.most_common(15):
            category_info = self.categorize_keyword_contextual(term, original_text)
            relevance_score = self.calculate_relevance_score(term, freq, total_relevant_words, category_info['category'])
            
            all_terms.append({
                'term': term,
                'type': 'trigram',
                'frequency': freq,
                'relevance_score': relevance_score,
                'category': category_info['category'],
                'detailed_category': category_info['detailed_category'],
                'classification_confidence': category_info['confidence'],
                'classification_method': category_info['method']
            })
        
        return all_terms, unigrams

    def categorize_terms(self, all_terms):
        keyword_insights = {
            'destination_terms': [],
            'timing_terms': [],
            'activity_terms': [],
            'price_terms': [],
            'sentiment_terms': [],
            'planning_terms': [],
            'accommodation_terms': [],
            'food_terms': [],
            'general_terms': []
        }
        
        for term_data in all_terms:
            category = term_data['category']
            keyword_insights[f'{category}_terms'].append(term_data)
        
        return keyword_insights

def test_single_site_with_transformers():
    # URL de teste
    url = "https://www.viagensmontreal.com/blog/quando-ir-a-porto-de-galinhas-descubra-a-melhor-epoca/"
    
    print("🚀 Testando Scraper com Transformers...")
    print(f"URL: {url}\n")
    
    # 1. Extrair conteúdo
    print("📥 Extraindo conteúdo...")
    import requests
    from bs4 import BeautifulSoup
    import re
    import nltk
    
    nltk.download('stopwords', quiet=True)
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.extract()
    
    text = soup.get_text()
    original_text = re.sub(r'\s+', ' ', text).strip()
    print(f"✅ Texto extraído: {len(original_text)} caracteres\n")
    
    # 2. Inicializar analyzer com transformers
    print("🤖 Carregando modelos transformers...")
    analyzer = TextAnalyzer()
    print("✅ Modelos carregados!\n")
    
    # 3. Análise de Sentimento com Transformer
    print("💭 Analisando sentimento com transformer...")
    sentiment_result = analyzer.analyze_sentiment_transformer(original_text)
    
    print("📊 RESULTADO DO SENTIMENTO:")
    print(f"  Score Geral: {sentiment_result['overall_sentiment']}")
    print(f"  Confiança: {sentiment_result['sentiment_confidence']}")
    print(f"  Chunks Analisados: {sentiment_result['chunks_analyzed']}")
    print(f"  Método: {sentiment_result['method']}\n")
    
    # 4. Preparar texto para análise de palavras
    print("🔤 Processando texto para análise de palavras...")
    
    stop_words = set(nltk.corpus.stopwords.words('portuguese'))
    stop_words.update(['que', 'para', 'com', 'uma', 'mais', 'muito', 'pode', 'ser', 'tem', 'vai', 'seu', 'sua'])
    
    clean_text = re.sub(r'[^\w\s]', ' ', original_text.lower())
    words = [word for word in clean_text.split() if len(word) > 2]
    filtered_words = [word for word in words if word not in stop_words]
    
    print(f"✅ {len(filtered_words)} palavras filtradas\n")
    
    # 5. Análise de N-gramas com categorização contextual
    print("🏷️ Categorizando termos com transformer...")
    
    all_terms, unigrams = analyzer.analyze_ngrams(filtered_words, len(filtered_words), original_text)
    
    print("📈 TOP 10 TERMOS COM CATEGORIZAÇÃO CONTEXTUAL:")
    
    for i, term in enumerate(all_terms[:10], 1):
        print(f"  {i}. '{term['term']}' ({term['type']})")
        print(f"     Categoria: {term['category']} -> {term['detailed_category']}")
        print(f"     Frequência: {term['frequency']} | Score: {term['relevance_score']}")
        print(f"     Confiança: {term['classification_confidence']} | Método: {term['classification_method']}")
        print()
    
    # 6. Categorizar termos
    print("📂 Organizando por categorias...")
    categorized = analyzer.categorize_terms(all_terms)
    
    print("📋 RESUMO POR CATEGORIA:")
    for category, terms in categorized.items():
        if terms:
            print(f"  {category.upper()}: {len(terms)} termos")
            # Mostrar os 3 mais relevantes de cada categoria
            top_terms = sorted(terms, key=lambda x: x['relevance_score'], reverse=True)[:3]
            for term in top_terms:
                print(f"    - {term['term']} (score: {term['relevance_score']})")
    
    print("\n" + "="*60)
    
    # 7. Comparação: Método antigo vs novo
    print("🔍 COMPARAÇÃO: Keyword vs Contextual")
    
    sample_terms = ["hospedagem", "hotéis", "melhor época", "pernambuco"]
    
    for term in sample_terms:
        matching_terms = [t for t in all_terms if term in t['term']]
        if matching_terms:
            term_data = matching_terms[0]
            
            # Método antigo (fallback)
            old_result = analyzer.categorize_keyword_fallback(term)
            
            print(f"\nTermo: '{term_data['term']}'")
            print(f"  Método Antigo: {old_result['category']}")
            print(f"  Método Novo: {term_data['category']} -> {term_data['detailed_category']}")
            print(f"  Confiança: {term_data['classification_confidence']}")
    
    print("\n" + "="*60)
    print("🎯 CONCLUSÃO:")
    print(f"✅ Sentimento geral do site: {sentiment_result['overall_sentiment']:.2f} ({'Positivo' if sentiment_result['overall_sentiment'] > 0.6 else 'Neutro' if sentiment_result['overall_sentiment'] > 0.4 else 'Negativo'})")
    print(f"✅ {len(all_terms)} termos categorizados automaticamente")
    print(f"✅ {len([t for t in all_terms if t['classification_method'] == 'contextual'])} termos categorizados com IA")
    print(f"✅ {len([t for t in all_terms if t['classification_method'] == 'keyword'])} termos categorizados com fallback")

if __name__ == "__main__":
    test_single_site_with_transformers()