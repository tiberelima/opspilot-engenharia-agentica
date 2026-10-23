"""Aula 4 — Observabilidade com Langfuse.

Por que observabilidade é o desafio nº 1 de agentes em produção?
---------------------------------------------------------------
Um agente toma decisões diferentes a cada execução. Quando algo dá errado
("por que ele abriu SEV1?", "por que custou 10x mais ontem?"), log de texto não
basta: você precisa do **trace** — a árvore completa de cada chamada ao modelo,
cada ferramenta, cada nó do grafo, com entradas, saídas, tokens, custo e latência.

O Langfuse (open source, SDK v4 baseado em OpenTelemetry) faz isso com duas peças:

* ``CallbackHandler`` — plugado no ``config`` de ``invoke``/``stream``, captura
  automaticamente tudo que o LangChain/LangGraph executa.
* ``propagate_attributes`` — carimba no trace ``session_id``, ``user_id``,
  ``tags`` e ``version`` para você filtrar no painel ("todas as execuções da
  versão 1.0 do prompt no plantão da Ana").

Tudo aqui é **opcional**: sem as chaves ``LANGFUSE_*`` o código roda normal,
só que sem trace. Agente nunca deve cair porque a observabilidade caiu.
"""

from __future__ import annotations

from typing import Any

from opspilot.config import LIMITE_RECURSAO, tem_langfuse
from opspilot.prompts import VERSAO_PROMPT
from opspilot.util import texto_de


def cliente_langfuse():
    """Retorna o cliente Langfuse configurado pelas variáveis de ambiente, ou ``None``."""
    if not tem_langfuse():
        return None
    from langfuse import get_client

    return get_client()


def invocar_com_trace(
    agente,
    pergunta: str,
    *,
    nome: str = "opspilot",
    session_id: str = "sessao-local",
    user_id: str = "aluno",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Executa o agente/grafo dentro de um trace do Langfuse.

    Args:
        agente: Qualquer grafo compilado (ReAct, FSM ou Deep Agent).
        pergunta: Mensagem do usuário.
        nome: Nome do trace no painel.
        session_id: Agrupa traces da mesma conversa/sessão.
        user_id: Quem disparou (pessoa do plantão, job de CI...).
        tags: Etiquetas livres para filtro (ex.: ``["aula4", "eval"]``).

    Returns:
        ``{"resposta": str, "estado": dict, "trace_id": str | None, "trace_url": str | None}``
    """
    langfuse = cliente_langfuse()
    config: dict[str, Any] = {"recursion_limit": LIMITE_RECURSAO}
    trace_id = trace_url = None

    if langfuse is None:
        estado = agente.invoke({"messages": [{"role": "user", "content": pergunta}]}, config=config)
        return {"resposta": texto_de(estado["messages"][-1]), "estado": estado, "trace_id": None, "trace_url": None}

    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    # ┌─ LACUNA [A4-L1] Plugue o Langfuse no LangChain/LangGraph
    # │  Dica: Crie um CallbackHandler() e coloque-o na lista config["callbacks"]. Tudo que rodar com esse config vira um trace automaticamente.
    # │  Copilot Chat: "Crie handler = CallbackHandler() e adicione em config["callbacks"] como uma lista"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=4`
    handler = CallbackHandler()
    config["callbacks"] = [handler]

    with langfuse.start_as_current_observation(as_type="span", name=nome, input={"pergunta": pergunta}) as span:
        # ┌─ LACUNA [A4-L2] Carimbe o trace com contexto de negócio
        # │  Dica: propagate_attributes(...) é um context manager. Passe session_id, user_id, tags e version=VERSAO_PROMPT — assim você filtra no painel por versão de prompt.
        # │  Copilot Chat: "Abra um with propagate_attributes passando session_id, user_id, tags (ou lista vazia) e version=VERSAO_PROMPT"
        # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=4`
        with propagate_attributes(session_id=session_id, user_id=user_id, tags=tags or [], version=VERSAO_PROMPT):
            estado = agente.invoke({"messages": [{"role": "user", "content": pergunta}]}, config=config)
        resposta = texto_de(estado["messages"][-1])
        span.update(output={"resposta": resposta})
        trace_id = langfuse.get_current_trace_id()
        trace_url = langfuse.get_trace_url(trace_id=trace_id)

    return {"resposta": resposta, "estado": estado, "trace_id": trace_id, "trace_url": trace_url}


def registrar_nota(trace_id: str | None, nome: str, valor: float, comentario: str = "") -> None:
    """Envia uma nota (score) de avaliação para o trace no Langfuse.

    Scores transformam traces em **métricas**: no painel dá para ver a média de
    "fundamentacao" por versão de prompt, por dia, por modelo...
    """
    langfuse = cliente_langfuse()
    if langfuse is None or trace_id is None:
        return
    # ┌─ LACUNA [A4-L5] Ligue a avaliação ao trace
    # │  Dica: langfuse.create_score(...) recebe trace_id, name, value (float) e comment.
    # │  Copilot Chat: "Chame langfuse.create_score com trace_id, name=nome, value=float(valor) e comment=comentario"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=4`
    langfuse.create_score(trace_id=trace_id, name=nome, value=float(valor), comment=comentario)


def finalizar() -> None:
    """Garante que os traces em buffer sejam enviados (importante em scripts curtos e no CI)."""
    langfuse = cliente_langfuse()
    if langfuse is not None:
        langfuse.flush()


__all__ = ["cliente_langfuse", "invocar_com_trace", "registrar_nota", "finalizar"]
