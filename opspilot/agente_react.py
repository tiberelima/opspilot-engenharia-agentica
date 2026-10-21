"""Aula 1 — O primeiro agente ReAct do OpsPilot.

Anatomia de um agente (o que você monta neste arquivo)
------------------------------------------------------
::

    ┌────────────┐   decide    ┌──────────────┐   executa   ┌──────────────┐
    │   MODELO   │ ──────────▶ │  FERRAMENTAS │ ──────────▶ │  OBSERVAÇÃO  │
    │  (cérebro) │ ◀────────── │   (corpo)    │             │  (resultado) │
    └────────────┘  lê o result└──────────────┘             └──────────────┘
          │
          └── quando não precisa de mais ferramentas → resposta final → PARSE

* **Modelo** — ``obter_llm()`` (OpenAI via ``init_chat_model``).
* **Ferramentas** — ``opspilot.tools``.
* **Prompt** — ``opspilot.prompts.montar_prompt_sistema()``.
* **Parse** — extrair a resposta final do estado retornado.

``create_agent`` (LangChain 1.x) monta esse ciclo para nós. Por baixo dos panos
ele já é um **grafo LangGraph** — na Aula 2 vamos abrir essa "caixa-preta" e
desenhar o grafo à mão.
"""

from __future__ import annotations

from langchain.agents import create_agent

from opspilot.config import LIMITE_RECURSAO, obter_llm
from opspilot.prompts import montar_prompt_sistema
from opspilot.tools import TODAS_FERRAMENTAS
from opspilot.util import texto_de


def criar_agente_react(ferramentas: list | None = None, checkpointer=None):
    """Cria o agente ReAct do OpsPilot.

    Args:
        ferramentas: Lista de ferramentas (padrão: todas do ``opspilot.tools``).
        checkpointer: Opcional. Com um checkpointer o agente lembra da conversa
            entre chamadas que usem o mesmo ``thread_id``.

    Returns:
        Um grafo compilado (``CompiledStateGraph``) com ``.invoke()`` e ``.stream()``.
    """
    # [A1-L5] ✔ Monte o agente: modelo + ferramentas + prompt
    agente = create_agent(
        model=obter_llm(),
        tools=ferramentas or TODAS_FERRAMENTAS,
        system_prompt=montar_prompt_sistema(),
        checkpointer=checkpointer,
        name="opspilot-react",
    )
    return agente


def perguntar(agente, pergunta: str, thread_id: str | None = None) -> str:
    """Envia uma pergunta ao agente e devolve só o texto da resposta final.

    Args:
        agente: Agente criado por :func:`criar_agente_react`.
        pergunta: Mensagem do usuário.
        thread_id: Identificador da conversa (necessário se houver checkpointer).

    Returns:
        O texto da última mensagem do agente.
    """
    config = {"recursion_limit": LIMITE_RECURSAO}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    resultado = agente.invoke({"messages": [{"role": "user", "content": pergunta}]}, config=config)

    # [A1-L6] ✔ Parse: extraia a resposta final do estado
    return texto_de(resultado["messages"][-1])
