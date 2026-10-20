"""Aula 2 — O OpsPilot como Máquina de Estados (FSM) em LangGraph.

Por que sair do ``create_agent`` e desenhar um grafo à mão?
-----------------------------------------------------------
O agente ReAct da Aula 1 decide TUDO sozinho: ótimo para explorar, ruim para
produção. Em produção queremos **controle de fluxo**:

* rotas diferentes por tipo de pedido (incidente, dúvida, fora de escopo);
* um **ciclo** de investigação com limite claro;
* **validação** do resultado e uma **rota de erro** com nova tentativa;
* **aprovação humana** antes de qualquer efeito colateral;
* **persistência**: poder pausar, desligar a máquina e retomar depois.

O grafo (gere o desenho com ``python -m aulas.aula2.a2_02_desenhar_grafo``)::

    START → classificar ─┬─ incidente ──▶ investigar ⇄ ferramentas
                         │                    │ (sem tool_calls)
                         │                    ▼
                         │                 validar ──erro & tentativas<3──▶ investigar
                         │                    │  └──erro & esgotou──▶ escalar_humano → END
                         │                    ▼ ok
                         │            aprovar_ticket (interrupt) ─sim─▶ abrir_ticket ─┐
                         │                    └─────────────nao─────────────────────┤
                         │                                                          ▼
                         ├─ duvida ─────▶ responder_duvida → END              responder → END
                         └─ fora_escopo ▶ recusar → END

Vocabulário LangGraph
    *State* (``EstadoOps``) é o "quadro branco" compartilhado. Cada **nó** é uma
    função ``estado -> atualização parcial``. **Arestas condicionais** são funções
    puras ``estado -> nome do próximo nó`` (por isso são fáceis de testar!).
"""

from __future__ import annotations

import re
from functools import lru_cache
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel, Field
from typing import Annotated, Literal, NotRequired, TypedDict

from opspilot.config import obter_llm
from opspilot.prompts import PROMPT_CLASSIFICADOR, PROMPT_EXTRATOR_DIAGNOSTICO, montar_prompt_sistema
from opspilot.tools import FERRAMENTAS_LEITURA, abrir_ticket, buscar_runbook, servicos_conhecidos, \
    notificar_canal
from opspilot.util import texto_de

#: Quantas vezes o validador pode mandar o agente investigar de novo.
MAX_TENTATIVAS = 3

#: Política de retry para falhas transitórias (timeout, 429, 5xx) em nós que chamam o LLM.
RETRY_LLM = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)


# --------------------------------------------------------------------------- #
# Saídas estruturadas (o "Parse" robusto: Pydantic em vez de regex no texto)
# --------------------------------------------------------------------------- #


class Classificacao(BaseModel):
    """Resultado do nó ``classificar``."""

    categoria: Literal["incidente", "duvida", "fora_escopo"]
    servico: str | None = Field(None, description="Serviço citado, se houver.")
    resumo: str = Field(description="O pedido em uma frase.")


class Diagnostico(BaseModel):
    """Diagnóstico que o nó ``validar`` extrai da investigação."""

    servico_causa_raiz: str = Field(description="Serviço onde está a causa raiz.")
    causa_raiz: str
    evidencias: list[str] = Field(description="Trechos de métricas/logs retornados pelas ferramentas.")
    sintomas: list[str] = Field(default_factory=list, description="Efeitos colaterais observados.")
    acao_recomendada: str
    severidade: Literal["SEV1", "SEV2", "SEV3"]


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #


class EstadoOps(TypedDict):
    """O "quadro branco" compartilhado entre os nós.

    Só ``messages`` é obrigatório: isso deixa o grafo ser usado como sub-agente
    de um Deep Agent na Aula 3 (que só envia ``{"messages": [...]}``).
    """

    # ┌─ LACUNA [A2-L1] O reducer que ACUMULA mensagens no estado
    # │  Dica: Sem reducer, cada nó que retorna "messages" SOBRESCREVE o histórico. Use Annotated[list[AnyMessage], <reducer>] — o reducer padrão do LangGraph para mensagens já está importado.
    # │  Copilot Chat: "Declare o campo messages como Annotated de list[AnyMessage] usando o reducer add_messages do LangGraph"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=2`
    messages: Annotated[list[AnyMessage], add_messages]
    categoria: NotRequired[str]
    servico: NotRequired[str | None]
    diagnostico: NotRequired[dict | None]
    tentativas: NotRequired[int]
    erros_validacao: NotRequired[list[str]]
    ticket_aprovado: NotRequired[bool]
    ticket: NotRequired[str | None]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _llm():
    """Cria o LLM só quando um nó precisa (o grafo pode ser desenhado/testado sem chave)."""
    return obter_llm()


def _pergunta_original(estado: EstadoOps) -> str:
    """Primeira mensagem humana do estado = o pedido do usuário."""
    for mensagem in estado["messages"]:
        if isinstance(mensagem, HumanMessage) and not texto_de(mensagem).startswith("[VALIDADOR]"):
            return texto_de(mensagem)
    return texto_de(estado["messages"][-1])


def _saidas_de_ferramentas(estado: EstadoOps) -> str:
    """Concatena tudo que as ferramentas devolveram (a "verdade" do caso)."""
    return "\n".join(texto_de(m) for m in estado["messages"] if isinstance(m, ToolMessage))


_TOKEN_COM_NUMERO = re.compile(r"[\w.=%:/-]*\d[\w.=%:/-]*")


def evidencia_ancorada(evidencia: str, saidas_ferramentas: str) -> bool:
    """Verifica se uma evidência cita algo que REALMENTE veio das ferramentas.

    Mitigação de alucinação barata e determinística: números (``4200``,
    ``default_pool_size=10``) ou termos longos (``PoolTimeout``) da evidência
    precisam aparecer nas saídas das ferramentas.
    """
    saidas = saidas_ferramentas.lower()
    tokens = [t.strip(".,:;") for t in _TOKEN_COM_NUMERO.findall(evidencia.lower())]
    tokens = [t for t in tokens if len(t) >= 2]
    if tokens:
        return any(t in saidas for t in tokens)
    palavras_longas = [p for p in re.findall(r"[\w-]{8,}", evidencia.lower())]
    return any(p in saidas for p in palavras_longas)


def validar_diagnostico(diag: Diagnostico, saidas_ferramentas: str) -> list[str]:
    """Regras de negócio do diagnóstico. Retorna a lista de problemas (vazia = ok).

    Função PURA: não chama LLM, então é testada em ``tests/test_grafo.py``.
    """
    problemas: list[str] = []
    if diag.servico_causa_raiz not in servicos_conhecidos():
        problemas.append(
            f"servico_causa_raiz '{diag.servico_causa_raiz}' não existe; use um de {servicos_conhecidos()}."
        )
    if len(diag.evidencias) < 2:
        problemas.append("São necessárias pelo menos 2 evidências.")
    nao_ancoradas = [e for e in diag.evidencias if not evidencia_ancorada(e, saidas_ferramentas)]
    if nao_ancoradas:
        problemas.append(f"Evidências sem respaldo nas ferramentas (possível alucinação): {nao_ancoradas}")
    if not diag.causa_raiz.strip():
        problemas.append("causa_raiz vazia.")
    return problemas


# --------------------------------------------------------------------------- #
# Nós
# --------------------------------------------------------------------------- #


def classificar(estado: EstadoOps) -> dict:
    """Decide o tipo de pedido com saída estruturada (sem 'adivinhar' texto)."""
    classificador = _llm().with_structured_output(Classificacao)
    resultado: Classificacao = classificador.invoke(
        [
            SystemMessage(PROMPT_CLASSIFICADOR.format(servicos=", ".join(servicos_conhecidos()))),
            HumanMessage(_pergunta_original(estado)),
        ]
    )
    return {"categoria": resultado.categoria, "servico": resultado.servico, "tentativas": 0}


def responder_duvida(estado: EstadoOps) -> dict:
    """Dúvida de procedimento: busca o runbook e responde SÓ com base nele (mini-RAG)."""
    pergunta = _pergunta_original(estado)
    contexto = buscar_runbook.invoke({"consulta": pergunta})
    resposta = _llm().invoke(
        [
            SystemMessage(
                "Responda à dúvida usando APENAS os runbooks abaixo. Cite o arquivo do runbook. "
                "Se o runbook não cobrir, diga isso.\n\n" + contexto
            ),
            HumanMessage(pergunta),
        ]
    )
    return {"messages": [AIMessage(texto_de(resposta))]}


def recusar(estado: EstadoOps) -> dict:
    """Fora do escopo: resposta fixa, sem gastar tokens."""
    return {
        "messages": [
            AIMessage(
                "Sou o OpsPilot e só ajudo com operação e incidentes da MercadoNimbus "
                "(métricas, logs, runbooks e tickets). Pode me contar o que está acontecendo?"
            )
        ]
    }


def investigar(estado: EstadoOps) -> dict:
    """Um passo do ciclo ReAct, agora como NÓ explícito do grafo."""
    modelo = _llm().bind_tools(FERRAMENTAS_LEITURA)
    instrucao = SystemMessage(
        montar_prompt_sistema()
        + "\n\nNesta etapa você SÓ investiga (não abra tickets). Quando tiver evidências "
          "suficientes, escreva o diagnóstico no formato pedido, sem chamar ferramentas."
    )
    resposta = modelo.invoke([instrucao, *estado["messages"]])
    return {"messages": [resposta]}


def validar(estado: EstadoOps) -> dict:
    """Extrai o diagnóstico estruturado e aplica as regras de validação."""
    tentativas = estado.get("tentativas", 0) + 1
    saidas = _saidas_de_ferramentas(estado)
    try:
        extrator = _llm().with_structured_output(Diagnostico)
        diag: Diagnostico = extrator.invoke(
            [
                SystemMessage(PROMPT_EXTRATOR_DIAGNOSTICO),
                *estado["messages"],
                HumanMessage("Extraia agora o diagnóstico estruturado da investigação acima."),
            ]
        )
        problemas = validar_diagnostico(diag, saidas)
    except Exception as erro:  # noqa: BLE001 — qualquer falha de parse vira rota de erro
        diag, problemas = None, [f"Falha ao estruturar o diagnóstico: {erro}"]

    atualizacao: dict = {
        "tentativas": tentativas,
        "erros_validacao": problemas,
        "diagnostico": diag.model_dump() if diag else None,
    }
    if problemas and tentativas < MAX_TENTATIVAS:
        # Feedback explícito para a próxima volta do ciclo (autocorreção guiada).
        atualizacao["messages"] = [
            HumanMessage(
                "[VALIDADOR] Seu diagnóstico foi reprovado:\n- "
                + "\n- ".join(problemas)
                + "\nInvestigue de novo usando as ferramentas e corrija."
            )
        ]
    return atualizacao


def aprovar_ticket(estado: EstadoOps) -> dict:
    """Human-in-the-loop: PAUSA o grafo e espera uma decisão humana.

    ``interrupt`` salva o estado no checkpointer e devolve o controle a quem chamou.
    A execução continua quando alguém chama o grafo com ``Command(resume=...)``.
    """
    diag = estado["diagnostico"] or {}
    # ┌─ LACUNA [A2-L5] Pause o grafo pedindo aprovação humana
    # │  Dica: interrupt(valor) pausa o grafo e mostra `valor` para quem chamou. Quando o grafo for retomado com Command(resume="sim"), interrupt(...) RETORNA "sim".
    # │  Copilot Chat: "Chame interrupt passando um dicionário com a pergunta "Abrir ticket?" e o rascunho do ticket (servico, severidade, causa_raiz) e guarde o retorno em decisao"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=2`
    decisao = interrupt(
        {
            "pergunta": "Abrir ticket com este diagnóstico? (sim/nao)",
            "rascunho": {k: diag.get(k) for k in ("servico_causa_raiz", "severidade", "causa_raiz")},
        }
    )
    return {"ticket_aprovado": str(decisao).strip().lower().startswith(("s", "y", "a"))}


def notificar(estado: EstadoOps) -> dict:
    """Efeito colateral controlado: só chega aqui depois de aprovação."""
    return notificar_canal.invoke(estado.get("ticket"))

def executar_abertura_ticket(estado: EstadoOps) -> dict:
    """Efeito colateral controlado: só chega aqui depois de aprovação."""
    diag = estado["diagnostico"]
    resultado = abrir_ticket.invoke(
        {
            "titulo": f"[OpsPilot] {diag['causa_raiz'][:80]}",
            "servico": diag["servico_causa_raiz"],
            "severidade": diag["severidade"],
            "resumo": diag["causa_raiz"] + " | Ação: " + diag["acao_recomendada"],
        }
    )
    return {"ticket": resultado}


def responder(estado: EstadoOps) -> dict:
    """Monta a resposta final a partir do diagnóstico VALIDADO (sem nova chamada ao LLM)."""
    d = estado["diagnostico"]
    evidencias = "\n".join(f"- {e}" for e in d["evidencias"])
    sintomas = "\n".join(f"- {s}" for s in d.get("sintomas") or []) or "- (nenhum relevante)"
    ticket = estado.get("ticket") or "Nenhum ticket aberto."
    texto = (
        f"**Causa raiz ({d['servico_causa_raiz']}):** {d['causa_raiz']}\n\n"
        f"**Evidências:**\n{evidencias}\n\n**Sintomas / efeitos colaterais:**\n{sintomas}\n\n"
        f"**Ação recomendada:** {d['acao_recomendada']}\n\n**Severidade:** {d['severidade']}\n\n"
        f"**Ticket:** {ticket}"
    )
    return {"messages": [AIMessage(texto)]}


def escalar_humano(estado: EstadoOps) -> dict:
    """Rota de erro final: falhar de forma segura e explicável."""
    problemas = "\n- ".join(estado.get("erros_validacao") or ["motivo desconhecido"])
    return {
        "messages": [
            AIMessage(
                f"⚠️ Não consegui um diagnóstico confiável após {estado.get('tentativas', 0)} tentativas.\n"
                f"Problemas encontrados:\n- {problemas}\n\nEscalando para o plantão humano."
            )
        ]
    }


# --------------------------------------------------------------------------- #
# Arestas condicionais (funções PURAS → testáveis sem LLM)
# --------------------------------------------------------------------------- #


def rota_por_categoria(estado: EstadoOps) -> Literal["investigar", "responder_duvida", "recusar"]:
    """Primeira bifurcação do grafo."""
    # ┌─ LACUNA [A2-L2] Roteie pela categoria
    # │  Dica: Mapeie cada categoria para o nome do NÓ seguinte: incidente→investigar, duvida→responder_duvida; qualquer outra coisa→recusar (o default é a rota segura).
    # │  Copilot Chat: "Retorne o nome do próximo nó com base em estado.get("categoria"): incidente -> "investigar", duvida -> "responder_duvida", caso contrário "recusar""
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=2`
    destinos = {"incidente": "investigar", "duvida": "responder_duvida"}
    return destinos.get(estado.get("categoria"), "recusar")


def rota_investigacao(estado: EstadoOps) -> Literal["ferramentas", "validar"]:
    """O CICLO: se o modelo pediu ferramenta, executa; senão, vai validar."""
    ultima = estado["messages"][-1]
    # ┌─ LACUNA [A2-L3] Feche o ciclo ReAct no grafo
    # │  Dica: Uma AIMessage que pede ferramenta tem a lista `tool_calls` não vazia. Use getattr(ultima, "tool_calls", None) para não quebrar com outros tipos de mensagem.
    # │  Copilot Chat: "Retorne "ferramentas" se a última mensagem tiver tool_calls não vazio, senão "validar""
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=2`
    return "ferramentas" if getattr(ultima, "tool_calls", None) else "validar"


def rota_validacao(estado: EstadoOps) -> Literal["investigar", "escalar_humano", "aprovar_ticket", "responder"]:
    """Rota de ERRO com retry limitado + rota feliz com ou sem ticket."""
    problemas = estado.get("erros_validacao") or []
    # ┌─ LACUNA [A2-L4] Rota de erro resiliente
    # │  Dica: Se há problemas: volte para "investigar" enquanto tentativas < MAX_TENTATIVAS; senão "escalar_humano". Sem problemas: SEV1/SEV2 vão para "aprovar_ticket", SEV3 vai direto para "responder".
    # │  Copilot Chat: "Implemente a rota: com problemas e tentativas < MAX_TENTATIVAS -> "investigar"; com problemas e tentativas esgotadas -> "escalar_humano"; sem problemas e severidade SEV1/SEV2 -> "aprovar_ticket"; senão "responder""
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=2`
    if problemas:
        return "investigar" if estado.get("tentativas", 0) < MAX_TENTATIVAS else "escalar_humano"
    severidade = (estado.get("diagnostico") or {}).get("severidade")
    return "aprovar_ticket" if severidade in ("SEV1", "SEV2") else "responder"


def rota_aprovacao(estado: EstadoOps) -> Literal["abrir_ticket", "responder"]:
    """Depois da decisão humana."""
    return "abrir_ticket" if estado.get("ticket_aprovado") else "responder"


# --------------------------------------------------------------------------- #
# Montagem do grafo
# --------------------------------------------------------------------------- #


def construir_grafo(aprovacao_humana: bool = True, checkpointer=None):
    """Monta e compila a FSM do OpsPilot.

    Args:
        aprovacao_humana: Se ``False``, o grafo nunca abre ticket (pula o
            ``interrupt``). Usado quando o grafo vira sub-agente na Aula 3.
        checkpointer: Onde salvar o estado a cada passo (``InMemorySaver``,
            ``SqliteSaver``...). Obrigatório se ``aprovacao_humana=True``,
            porque ``interrupt`` precisa persistir o estado para retomar.

    Returns:
        ``CompiledStateGraph`` pronto para ``invoke``/``stream``.
    """
    g = StateGraph(EstadoOps)

    # Nós que chamam o LLM ganham retry automático para falhas transitórias.
    g.add_node("classificar", classificar, retry_policy=RETRY_LLM)
    g.add_node("investigar", investigar, retry_policy=RETRY_LLM)
    g.add_node("ferramentas", ToolNode(FERRAMENTAS_LEITURA))
    g.add_node("validar", validar, retry_policy=RETRY_LLM)
    g.add_node("responder_duvida", responder_duvida, retry_policy=RETRY_LLM)
    g.add_node("recusar", recusar)
    g.add_node("responder", responder)
    g.add_node("escalar_humano", escalar_humano)

    g.add_edge(START, "classificar")
    g.add_conditional_edges("classificar", rota_por_categoria)
    g.add_conditional_edges("investigar", rota_investigacao)
    g.add_edge("ferramentas", "investigar")  # ← a aresta que cria o CICLO

    if aprovacao_humana:
        g.add_node("aprovar_ticket", aprovar_ticket)
        g.add_node("abrir_ticket", executar_abertura_ticket)
        g.add_node("notificar", notificar)

        g.add_conditional_edges("validar", rota_validacao)
        g.add_conditional_edges("aprovar_ticket", rota_aprovacao)

        g.add_edge("abrir_ticket", "notificar")
        g.add_edge("notificar", "responder")

    else:
        # Sem humano no loop: sucesso vai direto para a resposta.
        def rota_validacao_sem_ticket(estado: EstadoOps) -> Literal["investigar", "escalar_humano", "responder"]:
            destino = rota_validacao(estado)
            return "responder" if destino == "aprovar_ticket" else destino

        g.add_conditional_edges("validar", rota_validacao_sem_ticket)

    for final in ("responder", "responder_duvida", "recusar", "escalar_humano"):
        g.add_edge(final, END)

    # ┌─ LACUNA [A2-L6] Compile com persistência
    # │  Dica: compile() aceita checkpointer=... É ele que permite pausar no interrupt e retomar depois (até em outro processo, com SqliteSaver).
    # │  Copilot Chat: "Compile o grafo g passando o checkpointer recebido e o nome "opspilot-fsm""
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=2`
    return g.compile(checkpointer=checkpointer, name="opspilot-fsm")
