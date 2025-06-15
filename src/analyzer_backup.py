import torch
from collections import Counter
from transformers import pipeline
from sklearn.cluster import Birch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gc

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
        # Configuração GPU
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"\n🎮 Informações da GPU:")
            print(f"   - Memória Total: {gpu_info['total']:.0f}MB")
            print(f"   - Memória Livre: {gpu_info['free']:.0f}MB")
            print(f"   - Memória Em Uso: {gpu_info['used']:.0f}MB")
            print(f"   - Memória Cache: {gpu_info['cached']:.0f}MB\n")
        
        # Configuração do dispositivo
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU disponível: {gpu_name}")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            print("💻 GPU não encontrada, usando CPU")

        # Inicializar Birch clustering para análise de sentimentos
        self.birch_model = Birch(n_clusters=3, threshold=0.5)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.sentiment_clusters_fitted = False
        
        # Modelo de transformers como backup
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=self.device
            )
            print(f"✅ Modelo de sentimento multilíngue carregado! (Usando {self.device})")
        except Exception as e:
            print(f"❌ Erro no modelo de sentimento: {e}")
            self.sentiment_pipeline = None
            
        # Classificador contextual
        try:
            self.classifier_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            print(f"✅ Modelo de classificação carregado! (Usando {self.device})")
        except Exception as e:
            print(f"❌ Erro no modelo de classificação: {e}")
            self.classifier_pipeline = None
            
        # Keywords e categorias para análise tradicional (fallback)
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
        
        self.contextual_categories = [
            "destinos e lugares turísticos",
            "datas e períodos de viagem", 
            "atividades de lazer e turismo",
            "valores monetários e preços",
            "opiniões e avaliações",
            "organização de viagens",
            "hotéis e onde ficar",
            "comida e restaurantes"
        ]
                device=self.device
            )
            print(f"✅ Modelo de classificação carregado! (Usando {self.device})")
        except Exception as e:
            print(f"❌ Erro no modelo de classificação: {e}")
            self.classifier_pipeline = None

        # Dicionário expandido de palavras e expressões negativas em português
        # Categorizado por tipos de problemas comuns em turismo
        self.negative_words = {
            # Problemas gerais
            'não', 'ruim', 'péssimo', 'horrível', 'terrível', 'problema',
            'decepção', 'decepcionante', 'mal', 'pior', 'pessimo', 'negativo',
            'desagradável', 'insatisfeito', 'insatisfação', 'reclamação',
            'dificuldade', 'falha', 'erro', 'preocupante', 'desorganizado',
            
            # Problemas específicos de turismo
            'sujo', 'suja', 'imundo', 'imunda', 'fedorento', 'fedida',
            'caro', 'cara', 'superfaturado', 'absurdo', 'preço abusivo',
            'lotado', 'lotada', 'cheio', 'cheia', 'superlotado',
            'perigoso', 'perigosa', 'inseguro', 'insegura', 'assalto',
            'violência', 'roubo', 'assaltado', 'furtado',
            
            # Experiência turística
            'decepcionante', 'decepcionado', 'decepciona', 'decepcionou',
            'não vale', 'não valeu', 'não compensa', 'desperdício',
            'perda de tempo', 'perda de dinheiro', 'arrependido',
            'arrependida', 'desisti', 'desistir', 'cancelei',
            
            # Infraestrutura
            'quebrado', 'quebrada', 'estragado', 'estragada',
            'manutenção', 'reforma', 'obras', 'interditado',
            'fechado', 'fechada', 'sem estrutura', 'mal conservado',
            'precário', 'precária', 'abandono', 'abandonado',
            
            # Serviços
            'mal atendimento', 'péssimo atendimento', 'grosseiro',
            'grosseira', 'mal educado', 'mal educada', 'despreparado',
            'despreparada', 'demorado', 'demorada', 'lento', 'lenta',
            'atraso', 'atrasado', 'cancelado', 'cancelamento',
            
            # Hospedagem
            'barulho', 'barulhento', 'incômodo', 'incomoda',
            'desconfortável', 'desconforto', 'quarto ruim',
            'cama ruim', 'sem ar', 'ar quebrado', 'sem água',
            'água fria', 'infiltração', 'mofo', 'mofado',
            
            # Alimentação
            'comida ruim', 'mal cozido', 'frio', 'passado',
            'estragada', 'intoxicação', 'passou mal', 'dor de barriga',
            'caro demais', 'preço alto', 'quantidade pequena',
            
            # Transporte
            'atraso', 'atrasado', 'cancelado', 'lotação',
            'quebrou', 'enguiçou', 'problema mecânico', 'acidente',
            'trânsito', 'congestionamento', 'difícil acesso',
            
            # Clima/Ambiente
            'chuva', 'chuvoso', 'tempo ruim', 'clima ruim',
            'frio demais', 'calor demais', 'nublado', 'temporal',
            'ventania', 'poluído', 'poluição', 'sujo', 'sujeira',
            
            # Expressões compostas
            'não gostei', 'não recomendo', 'não voltaria',
            'nunca mais', 'péssima experiência', 'má experiência',
            'não vale a pena', 'deixa a desejar', 'muito caro',
            'preço abusivo', 'falta de', 'sem estrutura',
            'pouca opção', 'poucas opções', 'mal organizado',
            'mal planejado', 'sem planejamento', 'sem manutenção'
        }

    def categorize_keyword_contextual(self, term, context_text=""):
        """Categoriza usando contexto NEUTRO e categorias DISTINTAS"""
        
        if not self.classifier_pipeline:
            return self.categorize_keyword_fallback(term)
        
        # CONTEXTO MELHORADO - neutro + expandido
        analysis_text = f"Esta palavra ou frase '{term}' em contexto de viagem se refere a: {context_text[:500]}"
        
        try:
            result = self.classifier_pipeline(analysis_text, self.contextual_categories)
            
            # MAPEAMENTO MELHORADO - categorias mais distintas
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
            
            # THRESHOLD AJUSTADO para maior precisão
            if confidence > 0.5:
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
        """Método fallback EXPANDIDO com acomodação"""
        if any(keyword in term.lower() for keyword in self.destination_keywords):
            category = 'destination'
        elif any(keyword in term.lower() for keyword in self.timing_keywords):
            category = 'timing'
        elif any(keyword in term.lower() for keyword in self.activity_keywords):
            category = 'activity'
        elif any(keyword in term.lower() for keyword in self.price_keywords):
            category = 'price'
        elif any(keyword in term.lower() for keyword in self.accommodation_keywords):  # NOVO
            category = 'accommodation'
        else:
            category = 'general'
            
        return {
            'category': category,
            'detailed_category': category,
            'confidence': 0.7,
            'method': 'keyword'
        }

    def categorize_keyword(self, term, frequency, context_text=""):
        """Interface compatível com código antigo"""
        result = self.categorize_keyword_contextual(term, context_text)
        return result['category']

    def analyze_sentiment_transformer(self, text):
        if not self.sentiment_pipeline:
            return self.calculate_sentiment_fallback(text)
            
        # Verifica palavras negativas no texto completo
        text_lower = text.lower()
        negative_words_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # CHUNKS MENORES para BERT português com batch processing
        max_length = 400
        text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        chunk_sentiments = []
        
        # Filtrar chunks muito pequenos
        valid_chunks = [chunk for chunk in text_chunks if len(chunk.strip()) >= 20]
        
        # Aplica penalidade baseada em palavras negativas encontradas (muito suave)
        negative_penalty = min(0.10, negative_words_count * 0.02)  # Máximo de 0.10 de penalidade (era 0.15)
        
        if not valid_chunks:
            return self.calculate_sentiment_fallback(text)
        
        try:
            # Processamento em batch para GPU
            batch_size = 8 if torch.cuda.is_available() else 2
            for i in range(0, len(valid_chunks), batch_size):
                batch = valid_chunks[i:i + batch_size]
                
                # Limpa cache GPU se necessário
                if torch.cuda.is_available() and i % 32 == 0:
                    torch.cuda.empty_cache()
                
                results = self.sentiment_pipeline(batch, batch_size=batch_size)
                
                # Processa resultados do batch
                for result in results:
                    if isinstance(result, dict):
                        result = [result]
                        
                    # MAPEAMENTO ROBUSTO para diferentes modelos com pesos suaves
                    if 'POSITIVE' in result[0]['label'].upper() or 'LABEL_2' in result[0]['label']:
                        score = result[0]['score'] * 1.0  # Peso normal para positivos (era 0.9)
                    elif 'NEGATIVE' in result[0]['label'].upper() or 'LABEL_0' in result[0]['label']:
                        score = 1 - (result[0]['score'] * 0.95)  # Redução mínima para negativos (era 0.85)
                    else:  # Neutral
                        score = 0.5
                    
                    # Aplica penalidade de palavras negativas
                    adjusted_score = max(0.0, score - negative_penalty)
                    chunk_sentiments.append(adjusted_score)
                
        except Exception as e:
            print(f"Erro na análise de sentimento em batch: {e}")
            # Tenta processar um por um em caso de erro
            for chunk in valid_chunks:
                try:
                    result = self.sentiment_pipeline(chunk)
                    if 'POSITIVE' in result[0]['label'].upper() or 'LABEL_2' in result[0]['label']:
                        score = result[0]['score']
                    elif 'NEGATIVE' in result[0]['label'].upper() or 'LABEL_0' in result[0]['label']:
                        score = 1 - result[0]['score']
                    else:
                        score = 0.5
                    chunk_sentiments.append(score)
                except Exception as chunk_e:
                    print(f"Erro no chunk individual: {chunk_e}")
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
        """Análise de sentimento OTIMIZADA - permite positivos reais mas detecta contextos negativos críticos"""
        
        # Palavras positivas MUITO específicas para turismo
        strong_positive_words = {
            'excelente', 'incrível', 'maravilhoso', 'fantástico', 'espetacular', 
            'paradisíaco', 'imperdível', 'vale muito a pena', 'super recomendo',
            'adorei', 'amei', 'perfeito', 'sensacional', 'lindo demais',
            'experiência única', 'voltaria sempre', 'indico fortemente'
        }
        
        # Palavras positivas moderadas
        moderate_positive_words = {
            'bom', 'boa', 'ótimo', 'legal', 'bacana', 'interessante',
            'recomendo', 'vale a pena', 'gostei', 'bonito', 'bonita',
            'tranquilo', 'agradável', 'satisfeito', 'custo benefício',
            'bem localizado', 'organizado', 'limpo', 'seguro'
        }
        
        # Palavras ALTAMENTE críticas (só essas causam score baixo)
        highly_critical_words = {
            'péssimo', 'horrível', 'terrível', 'não recomendo', 'não vale a pena',
            'decepção total', 'nunca mais', 'desperdício', 'perda de tempo',
            'arrependido', 'péssima experiência', 'muito caro', 'preço abusivo',
            'perigoso', 'inseguro', 'sujo demais', 'horrível experiência'
        }
        
        text_lower = text.lower()
        
        # Análise de contexto negativo inteligente
        context_analysis = self.analyze_negative_context(text)
        
        # Contagens específicas
        strong_positive_count = sum(text_lower.count(word) for word in strong_positive_words)
        moderate_positive_count = sum(text_lower.count(word) for word in moderate_positive_words)
        highly_critical_count = sum(text_lower.count(word) for word in highly_critical_words)
        
        # Sistema de pontuação mais sensível
        strong_positive_score = strong_positive_count * 3.0    # Peso alto para muito positivo
        moderate_positive_score = moderate_positive_count * 1.5 # Peso moderado
        critical_negative_score = highly_critical_count * 2.0   # Peso alto só para críticas
        context_negative_score = context_analysis['high_freq_negative_context'] * 1.0
        
        total_positive_score = strong_positive_score + moderate_positive_score
        total_negative_score = critical_negative_score + context_negative_score
        total_evidence = total_positive_score + total_negative_score
        
        if total_evidence > 0:
            # Cálculo principal: proporção de evidências positivas
            base_score = total_positive_score / total_evidence
            
            # Boost para conteúdo claramente positivo
            if strong_positive_count > 0 and highly_critical_count == 0:
                base_score = min(1.0, base_score + 0.1)  # Boost para muito positivo
            
            # Penalização moderada para contexto negativo crítico
            if context_analysis['context_weight'] > 0.08:  # Threshold mais alto
                base_score = base_score - (context_analysis['context_weight'] * 0.3)  # Penalização menor
                
            # Garantir que sites realmente positivos sejam classificados como tal
            if total_positive_score > total_negative_score * 2:  # Evidência forte de positividade
                base_score = max(base_score, 0.6)  # Garante classificação positiva
                
            confidence = min(0.9, total_evidence / 3)
            
        else:
            # Sem evidências claras - assume neutro tendendo ao positivo (bias turístico)
            base_score = 0.58  # Neutro-positivo por padrão
            confidence = 0.3
            
        # Só força negativo em casos extremamente críticos
        if highly_critical_count > total_positive_score and context_analysis['context_weight'] > 0.1:
            base_score = min(base_score, 0.3)  # Força negativo apenas em casos extremos
            
        # Garantir limites
        final_score = max(0.0, min(1.0, base_score))
            
        return {
            'overall_sentiment': round(final_score, 3),
            'sentiment_confidence': round(confidence, 3),
            'chunks_analyzed': 1,
            'method': 'fallback_optimized_v2',
            'details': {
                'strong_positive_words': strong_positive_count,
                'moderate_positive_words': moderate_positive_count,
                'highly_critical_words': highly_critical_count,
                'high_freq_negative_context': context_analysis['high_freq_negative_context'],
                'context_patterns': context_analysis['found_patterns'],
                'context_weight': context_analysis['context_weight'],
                'positive_score': total_positive_score,
                'negative_score': total_negative_score,
                'final_score': round(final_score, 3)
            }
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
        """Análise com CONTEXTO EXPANDIDO para melhor categorização"""
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
        
        # Unigrams com contexto EXPANDIDO
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
        
        # Bigrams com contexto EXPANDIDO
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
        
        # Trigrams com contexto EXPANDIDO
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
            'accommodation_terms': [],  # NOVO
            'food_terms': [],
            'general_terms': []
        }
        
        for term_data in all_terms:
            category = term_data['category']
            keyword_insights[f'{category}_terms'].append(term_data)
        
        return keyword_insights

    def analyze_negative_context(self, text):
        """Analisa contexto negativo focando em palavras de alta frequência em contexto crítico"""
        text_lower = text.lower()
        
        # Palavras de alta frequência que são REALMENTE problemáticas quando em contexto negativo
        high_frequency_negative_indicators = {
            # Críticas diretas (muito específicas)
            'não recomendo', 'não vale', 'não gostei', 'não voltaria', 'nunca mais',
            'não compensa', 'não vale a pena', 'péssima experiência', 'decepção total',
            'arrependimento', 'desperdício', 'perda de tempo', 'perda de dinheiro',
            
            # Problemas críticos de infraestrutura (alta frequência = problema sistêmico)
            'sempre sujo', 'sempre lotado', 'sempre caro', 'sempre perigoso',
            'constantemente', 'toda vez', 'sempre tem problema', 'frequentemente',
            
            # Padrões que indicam problemas sistêmicos
            'muita gente reclama', 'vários comentários', 'maioria das pessoas',
            'todo mundo fala', 'comum encontrar', 'geralmente acontece',
            
            # Expressões que indicam intensidade negativa
            'extremamente', 'totalmente', 'completamente', 'absolutamente'
        }
        
        # Contar APENAS indicadores realmente críticos
        negative_context_count = 0
        found_patterns = []
        
        for pattern in high_frequency_negative_indicators:
            if pattern in text_lower:
                count = text_lower.count(pattern)
                negative_context_count += count
                found_patterns.append(f"{pattern} ({count}x)")
        
        # Análise de palavras de alta frequência em contexto negativo
        # Identifica quando palavras comuns (porto, galinhas, praia) aparecem muitas vezes
        # junto com palavras negativas específicas
        common_tourism_words = ['porto', 'galinhas', 'praia', 'hotel', 'pousada', 'restaurante']
        specific_negatives = ['sujo', 'caro', 'perigoso', 'ruim', 'péssimo', 'problema']
        
        high_freq_negative_context = 0
        for tourism_word in common_tourism_words:
            tourism_count = text_lower.count(tourism_word)
            if tourism_count > 10:  # Palavra aparece mais de 10 vezes (alta frequência)
                # Verifica se há contexto negativo próximo
                for negative in specific_negatives:
                    if negative in text_lower:
                        # Penaliza apenas se há evidência clara de contexto negativo
                        negative_count = text_lower.count(negative)
                        if negative_count > 2:  # Palavra negativa aparece múltiplas vezes
                            high_freq_negative_context += 1
        
        return {
            'negative_context_count': negative_context_count,
            'high_freq_negative_context': high_freq_negative_context,
            'found_patterns': found_patterns,
            # Penalização muito mais suave - só penaliza casos realmente críticos
            'context_weight': min(0.15, (negative_context_count * 0.03) + (high_freq_negative_context * 0.02))
        }