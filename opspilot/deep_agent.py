"""Aula 3 — O OpsPilot "Comandante de Incidente" com o Deep Agent Harness.

O que é um *harness*?
---------------------
O LLM é o motor; o **harness** é o carro em volta dele: o que deixa o motor
fazer trabalho longo e complexo sem se perder. O pacote ``deepagents``
(``langchain-ai/deepagents``) entrega um harness pronto, inspirado no Claude Code,
apoiado em **4 pilares**:

1. **Planejamento** — ferramenta ``write_todos``: o agente escreve e atualiza um
   plano (lista de tarefas com status). Planejar explicitamente reduz "perder o fio".
2. **Sistema de arquivos** — ``ls``, ``read_file``, ``write_file``, ``edit_file``,
   ``glob``, ``grep``: o agente *descarrega* contexto em arquivos em vez de
   lotar a janela de contexto (token economics!).
3. **Sub-agentes** — ferramenta ``task``: delega uma tarefa a um agente com
   **contexto isolado** (só recebe a tarefa, devolve só o resultado). Várias
   delegações podem rodar em paralelo.
4. **Memória longa** — arquivos ``AGENTS.md`` carregados no prompt a cada
   execução e que o próprio agente pode atualizar.

E o grafo da Aula 2?
    ``create_deep_agent`` devolve um grafo LangGraph, e aceita outro grafo
    compilado como sub-agente (``CompiledSubAgent``). Por isso a FSM da Aula 2
    vira um "especialista" que o Comandante consulta para ter um diagnóstico
    VALIDADO. É assim que se encapsula um fluxo LangGraph num harness.

Mapa de arquivos que o agente enxerga (backend composto)::

    /                → workspace/          (leitura e escrita: notas, relatórios)
    /memoria/        → workspace/memoria/  (AGENTS.md — memória longa)
    /base/           → data/               (SOMENTE leitura: runbooks, logs, incidentes)
"""

from __future__ import annotations

from deepagents import CompiledSubAgent, FilesystemPermission, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend

from opspilot.config import DIR_DADOS, DIR_WORKSPACE, obter_llm
from opspilot.grafo import construir_grafo
from opspilot.tools import abrir_ticket, buscar_runbook, consultar_metricas, ler_logs

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


PROMPT_ORQUESTRADOR = """Você é o OpsPilot, o agente orquestrador de investigação de incidentes da MercadoNimbus.
Sua missão é conduzir investigações complexas de ponta a ponta e entregar um
post-mortem impecável, com cada afirmação apoiada em evidência.

## Como trabalhar
1. PLANEJE primeiro com `write_todos` (5 a 8 tarefas) e mantenha o status atualizado.
2. Leia o incidente em `/base/incidentes/<ID>.md` e o formato na sua memória.
3. DELEGUE em paralelo sempre que possível (várias chamadas `task` na mesma rodada):
   - `analista-de-logs`: UM serviço por delegação (ex.: checkout-api, pagamentos-db...).
   - `pesquisador-de-runbooks`: procedimento oficial do problema encontrado.
   - `triagem-fsm`: segunda opinião com diagnóstico VALIDADO por regras.
4. Salve o que cada sub-agente devolveu em `/notas/<assunto>.md` (não confie na memória de curto prazo).
5. Escreva o relatório em `/relatorios/postmortem-<ID>.md` no formato padrão da memória.
6. AUTOCORREÇÃO: releia o relatório, confira cada número contra `/notas/` e corrija com `edit_file`.
7. Se aprendeu algo reutilizável, acrescente UMA linha em "Lições aprendidas" de `/memoria/AGENTS.md`.
8. Por fim, abra o ticket com `abrir_ticket` (um humano vai aprovar).

## Regras
- Nunca invente números ou horários. Separe causa raiz de sintomas.
- `/base/` é somente leitura.
- Termine com um resumo de 5 linhas do que foi feito e o caminho do relatório.
"""

# --------------------------------------------------------------------------- #
# Sub-agentes
# --------------------------------------------------------------------------- #

# ┌─ LACUNA [A3-L1] Declare um sub-agente especialista
# │  Dica: Um SubAgent é um dicionário com name, description (o Comandante lê isso para decidir QUANDO delegar), system_prompt e tools. Dê a ele só as ferramentas de que precisa: ler_logs e consultar_metricas.
# │  Copilot Chat: "Complete o dicionário SubAgent analista-de-logs com uma description orientada a ação e tools=[ler_logs, consultar_metricas]"
# └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=3`
ANALISTA_DE_LOGS: SubAgent = {
    "name": "analista-de-logs",
    "description": (
        "Analisa logs e métricas de UM serviço e devolve linha do tempo, anomalias e "
        "linhas de log literais como evidência. Use uma delegação por serviço."
    ),
    "system_prompt": (
        "Você analisa UM serviço por vez. Use ler_logs (filtros ERROR, WARN, pool, deploy) e "
        "consultar_metricas. Devolva: linha do tempo UTC, anomalias vs. baseline e as linhas "
        "de log LITERAIS que sustentam cada conclusão. Máximo 15 linhas."
    ),
    "tools": [ler_logs, consultar_metricas],
}

PESQUISADOR_DE_RUNBOOKS: SubAgent = {
    "name": "pesquisador-de-runbooks",
    "description": (
        "Busca nos runbooks internos o procedimento oficial de diagnóstico, mitigação e "
        "prevenção para um problema já identificado."
    ),
    "system_prompt": (
        "Use buscar_runbook com termos específicos. Devolva os passos de mitigação e "
        "prevenção citando o nome do arquivo do runbook. Não invente procedimentos."
    ),
    "tools": [buscar_runbook],
}


def criar_subagente_triagem() -> CompiledSubAgent:
    """Empacota a FSM da Aula 2 como sub-agente (sem aprovação humana interna)."""
    # ┌─ LACUNA [A3-L2] Encapsule o grafo LangGraph da Aula 2 no harness
    # │  Dica: CompiledSubAgent = dicionário com name, description e runnable. O runnable é o grafo compilado; use construir_grafo(aprovacao_humana=False) porque quem pede aprovação agora é o Comandante.
    # │  Copilot Chat: "Retorne um CompiledSubAgent com name "triagem-fsm", uma description explicando que devolve diagnóstico validado por regras, e runnable=construir_grafo(aprovacao_humana=False)"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=3`
    return {
        "name": "triagem-fsm",
        "description": "Fluxo determinístico da Aula 2: classifica, investiga e devolve um diagnóstico VALIDADO por regras (com evidências checadas).",
        "runnable": construir_grafo(aprovacao_humana=False),
    }


# --------------------------------------------------------------------------- #
# Backend (o "disco" do agente) e montagem
# --------------------------------------------------------------------------- #


def criar_backend() -> CompositeBackend:
    """Roteia caminhos virtuais para pastas reais do repositório.

    ``virtual_mode=True`` faz de ``root_dir`` a raiz "/" do agente: ele não
    consegue sair dessa pasta (``../`` é bloqueado). Guarda-corpo essencial.
    """
    DIR_WORKSPACE.mkdir(parents=True, exist_ok=True)
    # ┌─ LACUNA [A3-L3] Monte o sistema de arquivos do agente
    # │  Dica: CompositeBackend(default=..., routes={...}). O default é um FilesystemBackend na DIR_WORKSPACE; a rota "/base/" aponta para outro FilesystemBackend na DIR_DADOS.
    # │  Copilot Chat: "Retorne um CompositeBackend cujo default é FilesystemBackend(root_dir=DIR_WORKSPACE, virtual_mode=True) e com a rota "/base/" para FilesystemBackend(root_dir=DIR_DADOS, virtual_mode=True)"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=3`
    return CompositeBackend(
        default=FilesystemBackend(root_dir=DIR_WORKSPACE, virtual_mode=True),
        routes={"/base/": FilesystemBackend(root_dir=DIR_DADOS, virtual_mode=True)},
    )


#: Regras de permissão avaliadas na ordem: a primeira que casar vence.
PERMISSOES = [
    FilesystemPermission(operations=["write"], paths=["/base/**"], mode="deny"),
]


def criar_deep_agent(checkpointer=None):
    """Cria o Comandante de Incidente.

    Args:
        checkpointer: Necessário para o ``interrupt_on`` (aprovação humana do
            ``abrir_ticket``) e para conversas com várias rodadas.

    Returns:
        Grafo LangGraph compilado — mesma interface ``invoke``/``stream`` das aulas anteriores.
    """
    # ┌─ LACUNA [A3-L4] Ligue os 4 pilares no create_deep_agent
    # │  Dica: Passe model, tools (consultar_metricas e abrir_ticket), system_prompt, subagents (os 3), backend, memory=["/memoria/AGENTS.md"], permissions, interrupt_on={"abrir_ticket": True} e checkpointer.
    # │  Copilot Chat: "Chame create_deep_agent com model=obter_llm(), tools=[consultar_metricas, abrir_ticket], system_prompt=PROMPT_COMANDANTE, subagents com ANALISTA_DE_LOGS, PESQUISADOR_DE_RUNBOOKS e criar_subagente_triagem(), backend=criar_backend(), memory=["/memoria/AGENTS.md"], permissions=PERMISSOES, interrupt_on={"abrir_ticket": True}, checkpointer e name "opspilot-comandante""
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=3`
    return create_deep_agent(
        model=obter_llm(),
        tools=[consultar_metricas, abrir_ticket],
        system_prompt=PROMPT_ORQUESTRADOR,
        subagents=[ANALISTA_DE_LOGS, PESQUISADOR_DE_RUNBOOKS, criar_subagente_triagem()],
        backend=criar_backend(),
        memory=["/memoria/AGENTS.md"],
        permissions=PERMISSOES,
        interrupt_on={"abrir_ticket": True},
        checkpointer=checkpointer,
        name="opspilot-orquestrador",
    )
