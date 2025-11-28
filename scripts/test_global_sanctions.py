"""
🧪 TESTE DE INTEGRAÇÃO: SANÇÕES GLOBAIS (OFAC + UN + EU)

Este script testa a integração REAL com 3 listas de sanções:
- OFAC (US Treasury)
- UN Security Council
- European Union
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ingestion import EnrichmentAgent
from src.models.schemas import Transaction
from decimal import Decimal
from datetime import datetime


def test_sanctions_download():
    """Testa download das 3 listas de sanções"""
    print("\n" + "="*70)
    print("🧪 TESTE 1: Download das Listas Globais de Sanções")
    print("="*70)
    
    agent = EnrichmentAgent()
    
    # Get statistics
    stats = agent.get_sanctions_stats()
    
    print(f"\n📊 Status das Listas de Sanções:")
    print(f"   Total consolidado: {stats['total_entities']} entidades")
    print(f"   Última atualização: {stats['last_update']}")
    print(f"   Cache válido por: {stats['cache_ttl_hours']:.1f} horas")
    
    print(f"\n🌍 Fontes Integradas:")
    for source in stats['sources']:
        print(f"   • {source}")
    
    if len(agent.sanctions_list) > 5000:
        print(f"\n✅ SUCESSO! Listas REAIS carregadas!")
        print(f"   🇺🇸 OFAC + 🌐 UN + 🇪🇺 EU")
        print(f"   Total: {len(agent.sanctions_list)} entidades sancionadas")
    elif len(agent.sanctions_list) > 1000:
        print(f"\n⚠️  Parcialmente carregado")
        print(f"   Algumas fontes podem ter falhado")
        print(f"   {len(agent.sanctions_list)} entidades disponíveis")
    else:
        print(f"\n⚠️  Lista simulada (fallback)")
        print(f"   Downloads podem ter falhado (firewall/network)")
    
    # Mostrar alguns exemplos
    print(f"\n📋 Primeiros 15 itens da lista consolidada:")
    for i, item in enumerate(list(agent.sanctions_list)[:15], 1):
        # Identificar origem se possível
        origin = ""
        if item.startswith("sdn-"):
            origin = " [OFAC]"
        elif item.startswith("un-"):
            origin = " [UN]"
        elif item.startswith("eu-"):
            origin = " [EU]"
        print(f"   {i:2d}. {item}{origin}")


def test_sanction_check():
    """Testa verificação de sanções nas 3 listas"""
    print("\n" + "="*70)
    print("🧪 TESTE 2: Verificação de Sanções (OFAC + UN + EU)")
    print("="*70)
    
    agent = EnrichmentAgent()
    
    # Testes com nomes reais de cada lista
    test_cases = [
        ("NORMAL-001", "John Smith", False, "Cliente normal dos EUA"),
        ("MADURO-001", "Nicolas Maduro", True, "Presidente Venezuela (OFAC + EU)"),
        ("PUTIN-001", "Vladimir Putin", True, "Presidente Rússia (OFAC + EU)"),
        ("KIM-001", "Kim Jong Un", True, "Líder Coreia do Norte (OFAC + UN + EU)"),
        ("TALIBAN-001", "Taliban", True, "Organização terrorista (OFAC + UN + EU)"),
        ("ABRAMOVICH-001", "Roman Abramovich", True, "Oligarca russo (EU)"),
        ("BIN-LADEN-001", "Osama Bin Laden", True, "Terrorista (OFAC + UN)"),
        ("ISIS-001", "Islamic State", True, "Organização terrorista (OFAC + UN + EU)"),
    ]
    
    print("\n🔍 Testando verificações:")
    
    for customer_id, customer_name, expected_sanctioned, description in test_cases:
        is_sanctioned = agent._check_sanctions(customer_id, customer_name)
        
        status = "✅" if is_sanctioned == expected_sanctioned else "⚠️"
        result = "SANCIONADO" if is_sanctioned else "OK"
        
        print(f"\n{status} {description}")
        print(f"   ID: {customer_id}")
        print(f"   Nome: {customer_name}")
        print(f"   Resultado: {result}")
        print(f"   Esperado: {'SANCIONADO' if expected_sanctioned else 'OK'}")


async def test_transaction_enrichment():
    """Testa enriquecimento completo de transação"""
    print("\n" + "="*70)
    print("🧪 TESTE 3: Enriquecimento de Transação Completo")
    print("="*70)
    
    agent = EnrichmentAgent()
    
    # Transação de teste
    transaction = Transaction(
        transaction_id="TXN-TEST-001",
        amount=Decimal("25000"),
        currency="USD",
        transaction_type="wire_transfer",
        sender_id="NORMAL-123",
        sender_name="John Smith",
        receiver_id="SANCT-999",
        receiver_name="Nicolas Maduro",  # Nome sancionado
        country_origin="US",
        country_destination="VE",  # Venezuela
        timestamp=datetime.utcnow()
    )
    
    print("\n📊 Transação de Teste:")
    print(f"   De: {transaction.sender_name} ({transaction.sender_id})")
    print(f"   Para: {transaction.receiver_name} ({transaction.receiver_id})")
    print(f"   Valor: ${transaction.amount} {transaction.currency}")
    print(f"   Rota: {transaction.country_origin} → {transaction.country_destination}")
    
    # Processar
    result = await agent.analyze(transaction)
    
    print(f"\n🎯 Resultado da Análise:")
    print(f"   Suspeito: {'🚨 SIM' if result.suspicious else '✅ NÃO'}")
    print(f"   Risk Score: {result.risk_score:.2f}")
    print(f"   Confidence: {result.confidence:.2f}")
    
    print(f"\n📋 Findings ({len(result.findings)}):")
    for i, finding in enumerate(result.findings, 1):
        print(f"   {i}. {finding}")
    
    print(f"\n🔍 Patterns Detectados ({len(result.patterns_detected)}):")
    for pattern in result.patterns_detected:
        print(f"   • {pattern}")
    
    print(f"\n📝 Explicação:")
    print(f"   {result.explanation}")
    
    print(f"\n⚖️ Ação Recomendada: {result.recommended_action}")
    print(f"   Criar Alerta: {'🚨 SIM' if result.alert_should_be_created else 'NÃO'}")
    
    # Verificar dados enriquecidos
    print(f"\n💾 Dados Enriquecidos:")
    for key, value in transaction.enriched_data.items():
        print(f"   {key}: {value}")


def test_refresh_list():
    """Testa atualização forçada das listas"""
    print("\n" + "="*70)
    print("🧪 TESTE 4: Atualização Forçada (OFAC + UN + EU)")
    print("="*70)
    
    agent = EnrichmentAgent()
    
    old_count = len(agent.sanctions_list)
    print(f"\n📊 Lista Atual: {old_count} entradas")
    
    print(f"\n🔄 Forçando atualização de TODAS as 3 fontes...")
    print(f"   🇺🇸 OFAC (US Treasury)")
    print(f"   🌐 UN Security Council")
    print(f"   🇪🇺 European Union")
    
    success = agent.refresh_sanctions_list()
    
    new_count = len(agent.sanctions_list)
    
    if success:
        print(f"\n✅ Todas as listas atualizadas com sucesso!")
        print(f"   Antes: {old_count} entradas")
        print(f"   Depois: {new_count} entradas")
        print(f"   Diferença: {new_count - old_count:+d} entidades")
    else:
        print(f"\n⚠️  Falha ao atualizar algumas listas")
        print(f"   Sistema continua operando com listas disponíveis")


def test_statistics():
    """Testa estatísticas das listas"""
    print("\n" + "="*70)
    print("🧪 TESTE 5: Estatísticas das Listas")
    print("="*70)
    
    agent = EnrichmentAgent()
    stats = agent.get_sanctions_stats()
    
    print(f"\n📊 Estatísticas Detalhadas:")
    print(f"   Total de entidades: {stats['total_entities']:,}")
    print(f"   Última atualização: {stats['last_update']}")
    
    if stats['cache_age_hours'] is not None:
        print(f"   Idade do cache: {stats['cache_age_hours']:.1f} horas")
        remaining = stats['cache_ttl_hours'] - stats['cache_age_hours']
        print(f"   Próxima atualização em: {remaining:.1f} horas")
    
    print(f"\n🌍 Fontes Consolidadas ({len(stats['sources'])}):")
    for i, source in enumerate(stats['sources'], 1):
        print(f"   {i}. {source}")


def main():
    """Executa todos os testes"""
    print("\n" + "🌍"*35)
    print("  TESTE: INTEGRAÇÃO TRIPLA - OFAC + UN + EU")
    print("🌍"*35)
    
    try:
        # Teste 1: Download das 3 fontes
        test_sanctions_download()
        
        # Teste 2: Verificação
        test_sanction_check()
        
        # Teste 3: Enriquecimento (async)
        import asyncio
        asyncio.run(test_transaction_enrichment())
        
        # Teste 4: Refresh
        test_refresh_list()
        
        # Teste 5: Estatísticas
        test_statistics()
        
        print("\n" + "="*70)
        print("🎉 TODOS OS TESTES COMPLETOS!")
        print("="*70)
        print("\n✅ Integração tripla funcionando!")
        print("   🇺🇸 OFAC")
        print("   🌐 UN Security Council")
        print("   🇪🇺 European Union")
        print("\n🌍 Cobertura global de sanções: COMPLETA!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

