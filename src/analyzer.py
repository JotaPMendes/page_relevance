import torch
from collections import Counter
from transformers import pipeline
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gc
import re

def get_gpu_memory_info():
    """Retorna informações sobre o uso de memória da GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_allocated = torch.cuda.memory_allocated(0)
        gpu_memory_cached = torch.cuda.memory_reserved(0)
        gpu_memory_free = gpu_memory - gpu_memory_allocated
        
        return {
            'total': gpu_memory / (1024**2),  # MB
            'used': gpu_memory_allocated / (1024**2),  # MB
            'cached': gpu_memory_cached / (1024**2),  # MB
            'free': gpu_memory_free / (1024**2)  # MB
        }
    return None

class TextAnalyzer:
    def __init__(self):
        print("🔄 Inicializando TextAnalyzer com Birch Clustering...")
        
        # Configuração GPU
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"🎮 GPU: {gpu_info['total']:.0f}MB total, {gpu_info['free']:.0f}MB livre")
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🎮 Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("💻 Usando CPU")

        # Birch clustering para análise de sentimentos
        self.birch_model = Birch(n_clusters=3, threshold=0.5, branching_factor=50)
        self.vectorizer = TfidfVectorizer(
            max_features=500, 
            stop_words=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.sentiment_clusters_fitted = False
        self.texts_for_training = []
        
        # Modelo transformer como backup
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=self.device
            )
            print("✅ Modelo de sentimento carregado!")
        except:
            self.sentiment_pipeline = None
            print("⚠️ Usando apenas Birch clustering")
            
        # Configurações para análise tradicional
        self.setup_keywords_and_categories()
        print("✅ TextAnalyzer inicializado com sucesso!")

    def setup_keywords_and_categories(self):
        """Configura keywords e categorias para análise tradicional"""
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

    def prepare_text_for_clustering(self, text):
        """Prepara texto para análise com Birch"""
        # Limpeza básica
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extrair características importantes para clustering
        features = []
        
        # Palavras positivas comuns
        positive_indicators = ['excelente', 'ótimo', 'maravilhoso', 'recomendo', 'adorei', 'perfeito', 'incrível']
        positive_count = sum(1 for word in positive_indicators if word in text)
        features.append(f"positive_words_{min(positive_count, 5)}")
        
        # Palavras negativas comuns
        negative_indicators = ['ruim', 'péssimo', 'não recomendo', 'decepção', 'problema', 'caro demais']
        negative_count = sum(1 for word in negative_indicators if word in text)
        features.append(f"negative_words_{min(negative_count, 5)}")
        
        # Intensidade emocional
        intensity_words = ['muito', 'super', 'extremamente', 'totalmente', 'completamente']
        intensity_count = sum(1 for word in intensity_words if word in text)
        features.append(f"intensity_{min(intensity_count, 3)}")
        
        # Comprimento do texto (indicador de engajamento)
        text_length_category = "short" if len(text) < 100 else "medium" if len(text) < 500 else "long"
        features.append(f"length_{text_length_category}")
        
        return " ".join(features) + " " + text[:200]  # Combina features + texto truncado

    def train_birch_model(self, texts):
        """Treina o modelo Birch com os textos fornecidos"""
        print(f"🔄 Treinando Birch com {len(texts)} textos...")
        
        # Preparar textos para clustering
        processed_texts = [self.prepare_text_for_clustering(text) for text in texts]
        
        try:
            # Vetorização TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Treinar Birch
            self.birch_model.fit(tfidf_matrix)
            self.sentiment_clusters_fitted = True
            
            # Analisar clusters formados
            labels = self.birch_model.labels_
            n_clusters = len(set(labels))
            
            print(f"✅ Birch treinado! {n_clusters} clusters identificados")
            
            # Mapear clusters para sentimentos
            self.map_clusters_to_sentiments(texts, labels)
            
        except Exception as e:
            print(f"❌ Erro no treinamento Birch: {e}")
            self.sentiment_clusters_fitted = False

    def map_clusters_to_sentiments(self, texts, labels):
        """Mapeia clusters para sentimentos baseado em análise heurística"""
        cluster_sentiments = {}
        
        for cluster_id in set(labels):
            cluster_texts = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
            
            # Análise heurística para determinar sentimento do cluster
            total_positive = 0
            total_negative = 0
            
            for text in cluster_texts:
                text_lower = text.lower()
                
                # Contar indicadores positivos
                positive_words = ['excelente', 'ótimo', 'maravilhoso', 'recomendo', 'adorei', 'perfeito', 'incrível', 'fantástico']
                positive_count = sum(1 for word in positive_words if word in text_lower)
                
                # Contar indicadores negativos
                negative_words = ['ruim', 'péssimo', 'não recomendo', 'decepção', 'problema', 'terrível', 'horrível']
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                total_positive += positive_count
                total_negative += negative_count
            
            # Determinar sentimento do cluster
            if total_positive > total_negative * 1.5:
                sentiment = "positive"
                sentiment_score = 0.7 + (total_positive / (len(cluster_texts) + 1)) * 0.2
            elif total_negative > total_positive * 1.5:
                sentiment = "negative"
                sentiment_score = 0.3 - (total_negative / (len(cluster_texts) + 1)) * 0.2
            else:
                sentiment = "neutral"
                sentiment_score = 0.5
            
            cluster_sentiments[cluster_id] = {
                'sentiment': sentiment,
                'score': max(0.1, min(0.9, sentiment_score)),
                'sample_size': len(cluster_texts)
            }
            
            print(f"   Cluster {cluster_id}: {sentiment} (score: {sentiment_score:.2f}, {len(cluster_texts)} textos)")
        
        self.cluster_sentiments = cluster_sentiments

    def calculate_sentiment_birch(self, text):
        """Calcula sentimento usando Birch clustering"""
        if not self.sentiment_clusters_fitted:
            return self.calculate_sentiment_fallback(text)
        
        try:
            # Preparar texto
            processed_text = self.prepare_text_for_clustering(text)
            
            # Vetorizar
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predizer cluster
            cluster_id = self.birch_model.predict(text_vector)[0]
            
            # Obter sentimento do cluster
            if cluster_id in self.cluster_sentiments:
                cluster_info = self.cluster_sentiments[cluster_id]
                return {
                    'overall_sentiment': cluster_info['score'],
                    'sentiment_confidence': 0.8,
                    'method': 'birch_clustering',
                    'cluster_id': int(cluster_id),
                    'cluster_sentiment': cluster_info['sentiment']
                }
            else:
                return self.calculate_sentiment_fallback(text)
                
        except Exception as e:
            print(f"❌ Erro na análise Birch: {e}")
            return self.calculate_sentiment_fallback(text)

    def calculate_sentiment_fallback(self, text):
        """Análise de sentimento simplificada como fallback"""
        text_lower = text.lower()
        
        # Palavras positivas
        positive_words = ['excelente', 'ótimo', 'maravilhoso', 'recomendo', 'adorei', 'perfeito', 'incrível', 'fantástico', 'lindo', 'show']
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Palavras negativas
        negative_words = ['ruim', 'péssimo', 'não recomendo', 'decepção', 'problema', 'terrível', 'horrível', 'caro demais']
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Cálculo simples
        if positive_count > negative_count:
            score = 0.6 + min(0.3, positive_count * 0.1)
        elif negative_count > positive_count:
            score = 0.4 - min(0.3, negative_count * 0.1)
        else:
            score = 0.5
        
        return {
            'overall_sentiment': round(max(0.1, min(0.9, score)), 3),
            'sentiment_confidence': 0.6,
            'method': 'fallback_simple',
            'positive_words': positive_count,
            'negative_words': negative_count
        }

    def calculate_sentiment(self, text):
        """Interface principal para cálculo de sentimento"""
        if self.sentiment_clusters_fitted:
            result = self.calculate_sentiment_birch(text)
        else:
            result = self.calculate_sentiment_fallback(text)
        
        return result['overall_sentiment']

    def analyze_sentiment_transformer(self, text):
        """Mantém compatibilidade com código existente"""
        return self.calculate_sentiment_birch(text) if self.sentiment_clusters_fitted else self.calculate_sentiment_fallback(text)

    # Métodos de categorização (mantidos do código original)
    def categorize_keyword_fallback(self, term):
        """Método fallback para categorização"""
        term_lower = term.lower()
        
        if any(keyword in term_lower for keyword in self.destination_keywords):
            category = 'destination'
        elif any(keyword in term_lower for keyword in self.timing_keywords):
            category = 'timing'
        elif any(keyword in term_lower for keyword in self.activity_keywords):
            category = 'activity'
        elif any(keyword in term_lower for keyword in self.price_keywords):
            category = 'price'
        elif any(keyword in term_lower for keyword in self.accommodation_keywords):
            category = 'accommodation'
        else:
            category = 'general'
            
        return {
            'category': category,
            'detailed_category': category,
            'confidence': 0.7,
            'method': 'keyword'
        }

    def categorize_keyword_contextual(self, term, context_text=""):
        """Categorização com contexto (simplificada)"""
        return self.categorize_keyword_fallback(term)

    def categorize_keyword(self, term, frequency, context_text=""):
        """Interface compatível"""
        result = self.categorize_keyword_fallback(term)
        return result['category']

    def calculate_relevance_score(self, term, frequency, total_words, category):
        """Cálculo de relevância"""
        base_score = (frequency / total_words) * 1000
        multiplier = self.category_multipliers.get(category, 1.0)
        relevance_score = base_score * multiplier
        return round(min(relevance_score, 10.0), 2)

    def analyze_ngrams(self, filtered_words, total_relevant_words, original_text=""):
        """Análise de n-gramas simplificada"""
        unigrams = Counter(filtered_words)
        
        # Bigrams
        bigrams = []
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
            bigrams.append(bigram)
        bigrams_count = Counter(bigrams)
        
        # Trigrams
        trigrams = []
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
            trigrams.append(trigram)
        trigrams_count = Counter(trigrams)
        
        all_terms = []
        
        # Processar unigrams
        for term, freq in unigrams.most_common(30):
            category_info = self.categorize_keyword_fallback(term)
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
        
        # Processar bigrams
        for term, freq in bigrams_count.most_common(20):
            category_info = self.categorize_keyword_fallback(term)
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
        
        # Processar trigrams
        for term, freq in trigrams_count.most_common(15):
            category_info = self.categorize_keyword_fallback(term)
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
        """Categorização de termos"""
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
