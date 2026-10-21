"""Aula 1 · Exercício 1 — LLM puro vs. agente.

Objetivo: SENTIR a diferença antes de programar o agente.

1. Perguntamos ao LLM "puro" (sem ferramentas) por que o checkout está lento.
   Ele não tem acesso a nada da empresa → vai *chutar* (ou admitir que não sabe).
2. Mostramos o mesmo modelo com ferramentas "amarradas" (``bind_tools``): em vez
   de responder, ele devolve um PEDIDO para chamar uma ferramenta. Esse pedido
   (``tool_calls``) é o átomo de todo agente.

Como rodar (no terminal do Codespace)::

    python -m aulas.aula1.a1_01_llm_vs_agente
"""

from __future__ import annotations

from opspilot.config import MODELO, obter_llm
from opspilot.tools import FERRAMENTAS_LEITURA
from opspilot.util import texto_de

PERGUNTA = "Por que o checkout da MercadoNimbus está lento agora?"


def main() -> None:
    print(f"Modelo em uso: {MODELO}\n")

    # [A1-L1] ✔ Seu primeiro contato com o LLM
    llm = obter_llm()
    resposta_pura = llm.invoke(PERGUNTA)
    print("=== 1) LLM puro (sem ferramentas) ===")
    print(texto_de(resposta_pura))

    # bind_tools anexa o JSON Schema das ferramentas à chamada do modelo.
    llm_com_ferramentas = llm.bind_tools(FERRAMENTAS_LEITURA)
    resposta_agente = llm_com_ferramentas.invoke(PERGUNTA)
    print("\n=== 2) Mesmo modelo, agora com ferramentas disponíveis ===")
    print("Texto:", texto_de(resposta_agente) or "(vazio — o modelo preferiu agir)")
    print("Pedidos de ferramenta (tool_calls):")
    for chamada in resposta_agente.tool_calls:
        print(f"  • {chamada['name']}({chamada['args']})")

    print(
        "\n💡 Repare: no passo 2 o modelo NÃO executa nada. Ele só pede. Quem executa "
        "a ferramenta e devolve o resultado é o *loop do agente* — próximo exercício."
    )


if __name__ == "__main__":
    main()
