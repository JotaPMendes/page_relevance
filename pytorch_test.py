import torch

print("🔍 VERIFICANDO PYTORCH E CUDA")
print("="*40)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Versão CUDA PyTorch: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    
    # Teste prático
    print("\n🧪 TESTE PRÁTICO")
    print("-"*20)
    
    try:
        # Criar tensor na GPU
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        
        print("✅ Tensor criado e operação executada na GPU!")
        print(f"✅ Device do tensor: {z.device}")
        
        # Limpar memória
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        
else:
    print("❌ CUDA não disponível")
    print("💡 Precisa reinstalar PyTorch com CUDA")
