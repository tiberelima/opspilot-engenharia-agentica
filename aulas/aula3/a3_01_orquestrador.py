"""Aula 3 · Exercício 1 — O Deep Agent investiga o INC-2041 e escreve o post-mortem.

O que observar enquanto roda:
    📋 o PLANO (``write_todos``) aparecendo e sendo atualizado;
    🧭 as DELEGAÇÕES (``task``) para os sub-agentes — várias na mesma rodada = paralelo;
    📁 os ARQUIVOS criados em ``workspace/notas`` e ``workspace/relatorios``;
    🧠 a MEMÓRIA ``workspace/memoria/AGENTS.md`` (rode ``git diff workspace/memoria`` no fim!);
    ✋ a APROVAÇÃO humana antes de ``abrir_ticket``.

Como rodar::

    python -m aulas.aula3.a3_01_orquestrador
    python -m aulas.aula3.a3_01_orquestrador --incidente INC-2041 --auto-aprovar

Custo: este exercício faz várias chamadas ao modelo (é um agente "profundo").
Com um modelo *mini* fica na casa de centavos de dólar por execução.
"""

from __future__ import annotations

import argparse
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from opspilot.config import DIR_WORKSPACE, LIMITE_RECURSAO
from opspilot.deep_agent import criar_deep_agent
from opspilot.util import texto_de

ICONES = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}


def mostrar_plano(todos: list[dict]) -> None:
    """Imprime a lista de tarefas que o agente mantém via ``write_todos``."""
    print("\n📋 PLANO DO AGENTE")
    for item in todos:
        print(f"   {ICONES.get(item.get('status'), '•')} {item.get('content')}")


def mostrar_mensagem(m) -> None:
    """Resumo de uma mensagem do agente principal (sub-agentes rodam isolados)."""
    for chamada in getattr(m, "tool_calls", None) or []:
        args = chamada["args"]
        if chamada["name"] == "task":
            alvo = args.get("subagent_type") or args.get("agent") or "?"
            print(f"🧭 delega → {alvo}: {str(args.get('description', ''))[:110]}")
        elif chamada["name"] == "write_todos":
            continue  # o plano é exibido separadamente
        else:
            caminho = args.get("file_path") or args.get("path") or ""
            print(f"🔧 {chamada['name']} {caminho}".rstrip())
    if m.type == "ai" and not getattr(m, "tool_calls", None):
        print(f"\n✅ RESPOSTA FINAL DO ORQUESTRADOR\n{texto_de(m)}")


def decidir(interrupcao: dict, auto_aprovar: bool) -> dict:
    """Converte a decisão humana para o formato do HumanInTheLoopMiddleware."""
    decisoes = []
    for acao in interrupcao["action_requests"]:
        print(f"\n✋ Aprovação necessária: {acao['name']}")
        for chave, valor in acao["args"].items():
            print(f"   • {chave}: {valor}")
        # [A3-L5] ✔ Responda ao humano-no-loop do harness
        resposta = "s" if auto_aprovar else input("   Aprovar? [s/n]: ").strip().lower()
        if resposta.startswith("s"):
            decisoes.append({"type": "approve"})
        else:
            decisoes.append({"type": "reject", "message": "Reprovado pelo plantão"})
    return {"decisions": decisoes}


def executar(agente, entrada, config) -> None:
    """Faz streaming do estado e mostra plano + ações conforme acontecem."""
    vistas, plano_anterior = 0, None
    for estado in agente.stream(entrada, config=config, stream_mode="values"):
        todos = estado.get("todos")
        if todos and todos != plano_anterior:
            mostrar_plano(todos)
            plano_anterior = todos
        mensagens = estado.get("messages", [])
        for m in mensagens[vistas:]:
            mostrar_mensagem(m)
        vistas = len(mensagens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incidente", default="INC-2041")
    parser.add_argument("--auto-aprovar", action="store_true", help="Aprova o ticket sem perguntar (CI/demo)")
    args = parser.parse_args()

    agente = criar_deep_agent(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"aula3-{uuid.uuid4().hex[:8]}"}, "recursion_limit": LIMITE_RECURSAO * 3}

    pedido = (
        f"Investigue o incidente {args.incidente} (detalhes em /base/incidentes/{args.incidente}.md) "
        f"e entregue o post-mortem em /relatorios/postmortem-{args.incidente}.md."
    )
    executar(agente, {"messages": [{"role": "user", "content": pedido}]}, config)

    # Enquanto houver ação aguardando aprovação humana, decide e retoma.
    snapshot = agente.get_state(config)
    while snapshot.interrupts:
        decisao = decidir(snapshot.interrupts[0].value, args.auto_aprovar)
        executar(agente, Command(resume=decisao), config)
        snapshot = agente.get_state(config)

    print("\n📁 Arquivos no workspace:")
    for caminho in sorted(DIR_WORKSPACE.rglob("*.md")):
        print(f"   {caminho.relative_to(DIR_WORKSPACE.parent)}")
    print("\n🧠 Dica: rode `git diff workspace/memoria` para ver o que o agente aprendeu.")


if __name__ == "__main__":
    main()