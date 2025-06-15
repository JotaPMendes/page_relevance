import torch
import sys
import os

def check_gpu_detailed():
    """Verificação detalhada da GPU"""
    print("🔍 DIAGNÓSTICO COMPLETO DA GPU")
    print("="*50)
    
    # 1. Verificar se CUDA está disponível
    print(f"✅ CUDA disponível: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA não está disponível. Possíveis causas:")
        print("   - PyTorch instalado sem suporte CUDA")
        print("   - Drivers NVIDIA não instalados")
        print("   - GPU não compatível")
        return False
    
    # 2. Informações da GPU
    print(f"🎮 Número de GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        print(f"   GPU {i}: {gpu_name} ({total_memory:.1f}GB)")
    
    # 3. GPU atual
    current_device = torch.cuda.current_device()
    print(f"🎯 GPU atual: {current_device}")
    
    # 4. Versões
    print(f"🔧 Versão PyTorch: {torch.__version__}")
    print(f"🔧 Versão CUDA: {torch.version.cuda}")
    
    # 5. Teste de memória
    try:
        allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
        cached = torch.cuda.memory_reserved(0) / (1024**2)  # MB
        print(f"💾 Memória alocada: {allocated:.1f}MB")
        print(f"💾 Memória cache: {cached:.1f}MB")
    except Exception as e:
        print(f"❌ Erro ao verificar memória: {e}")
    
    # 6. Teste prático
    print("\n🧪 TESTE PRÁTICO")
    print("-"*30)
    
    try:
        # Criar tensor na GPU
        test_tensor = torch.randn(1000, 1000).cuda()
        print("✅ Tensor criado na GPU com sucesso")
        
        # Operação simples
        result = test_tensor @ test_tensor.T
        print("✅ Operação matricial na GPU executada")
        
        # Limpar memória
        del test_tensor, result
        torch.cuda.empty_cache()
        print("✅ Memória limpa com sucesso")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste prático: {e}")
        return False

def test_transformers_gpu():
    """Testa se os transformers conseguem usar a GPU"""
    print("\n🤖 TESTE DOS TRANSFORMERS")
    print("-"*30)
    
    try:
        from transformers import pipeline
        
        # Criar pipeline de análise de sentimentos com modelo mais seguro
        print("📡 Carregando modelo de sentimentos multilíngue...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=0 if torch.cuda.is_available() else -1,  # 0 = GPU, -1 = CPU
            use_safetensors=True
        )
        
        device_used = "GPU" if sentiment_pipeline.device.type == "cuda" else "CPU"
        print(f"✅ Modelo carregado no dispositivo: {device_used}")
        
        # Teste de análise
        test_text = "Este produto é muito ruim e decepcionante."
        result = sentiment_pipeline(test_text)
        print(f"✅ Análise executada: {result}")
        
        # Teste de classificação zero-shot
        print("\n📊 Testando classificação zero-shot...")
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
            device=0 if torch.cuda.is_available() else -1,
            use_safetensors=True
        )
        
        test_labels = ["turismo", "tecnologia", "alimentação"]
        classification_result = classifier("Visitei Porto de Galinhas e a praia estava linda", test_labels)
        print(f"✅ Classificação executada: {classification_result['labels'][0]} ({classification_result['scores'][0]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste dos transformers: {e}")
          # Teste alternativo com modelo mais simples
        print("\n🔄 Tentando modelo alternativo...")
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
                # Removido use_safetensors=True pois este modelo já usa automaticamente
            )
            
            device_used = "GPU" if sentiment_pipeline.device.type == "cuda" else "CPU"
            print(f"✅ Modelo alternativo carregado no dispositivo: {device_used}")
            
            test_text = "Este produto é muito ruim e decepcionante."
            result = sentiment_pipeline(test_text)
            print(f"✅ Análise executada: {result}")
            
            return True
            
        except Exception as e2:
            print(f"❌ Erro no modelo alternativo: {e2}")
            return False

if __name__ == "__main__":
    gpu_ok = check_gpu_detailed()
    
    if gpu_ok:
        transformers_ok = test_transformers_gpu()
        
        if transformers_ok:
            print("\n🎉 TUDO FUNCIONANDO!")
            print("✅ GPU disponível e funcional")
            print("✅ Transformers usando GPU")
        else:
            print("\n⚠️ GPU OK, mas transformers com problemas")
    else:
        print("\n❌ Problemas com a GPU detectados")
        print("💡 Sugestões:")
        print("   1. Reinstalar PyTorch com CUDA")
        print("   2. Verificar drivers NVIDIA")
        print("   3. Usar CPU como alternativa")