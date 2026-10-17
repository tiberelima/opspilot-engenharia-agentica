"""Ferramentas (tools) do OpsPilot — o "corpo" do agente.

Conceito-chave da Aula 1
------------------------
Um LLM sozinho só *gera texto*. Ele vira **agente** quando pode *agir*: escolher
uma ferramenta, receber o resultado (observação) e decidir o próximo passo.
Isso é o ciclo **ReAct** = *Reason + Act*.

Para o LLM conseguir usar uma função Python como ferramenta, três coisas
importam MUITO mais do que o código em si:

1. **Nome** claro (é o "verbo" que o modelo vai escolher).
2. **Docstring** — vira a descrição que o modelo lê para decidir *quando* usar.
3. **Tipos dos parâmetros** — viram o JSON Schema que o modelo precisa preencher.

Boas práticas aplicadas aqui (e que valem no seu dia a dia):

* **Erros viram observação, não exceção.** Se o serviço não existe, devolvemos
  um texto explicando e listando opções válidas. O agente lê e se corrige
  sozinho — isso é *autocorreção* barata.
* **Saídas curtas e estruturadas.** Toda saída de ferramenta entra no contexto
  do modelo e custa tokens (*token economics*). Por isso há limites de linhas.
* **Ferramentas de leitura x de escrita.** ``abrir_ticket`` tem efeito colateral;
  na Aula 2 ela só roda depois de aprovação humana.

Todas as ferramentas leem a pasta ``data/`` (dados fictícios da MercadoNimbus).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Literal

from langchain_core.tools import tool

from opspilot.config import ARQ_METRICAS, DIR_LOGS, DIR_RUNBOOKS, DIR_TICKETS

# --------------------------------------------------------------------------- #
# Funções auxiliares (não são ferramentas: o LLM não as enxerga)
# --------------------------------------------------------------------------- #


def _carregar_metricas() -> dict:
    """Lê ``data/metricas.json`` ignorando chaves internas que começam com ``_``."""
    dados = json.loads(ARQ_METRICAS.read_text(encoding="utf-8"))
    return {nome: valores for nome, valores in dados.items() if not nome.startswith("_")}


def servicos_conhecidos() -> list[str]:
    """Lista os serviços monitorados. Útil para validação (Aula 2) e testes."""
    return sorted(_carregar_metricas())


def _normalizar(texto: str) -> list[str]:
    """Quebra um texto em palavras minúsculas com 3+ letras (busca simples)."""
    return [p for p in re.findall(r"[a-zà-ú0-9-]+", texto.lower()) if len(p) >= 3]


# --------------------------------------------------------------------------- #
# Ferramentas de LEITURA
# --------------------------------------------------------------------------- #


# ┌─ LACUNA [A1-L2] Transforme a função em uma ferramenta que o LLM enxerga
# │  Dica: O LangChain tem um decorator que lê o nome, a docstring e os tipos da função para gerar o schema da ferramenta. Ele já foi importado no topo do arquivo.
# │  Copilot Chat: "Transforme a função consultar_metricas em uma ferramenta LangChain usando o decorator importado de langchain_core.tools"
# └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=1`
@tool
def consultar_metricas(servico: str) -> str:
    """Consulta as métricas atuais de UM serviço e compara com o baseline.

    Use SEMPRE que precisar de números (latência, erros, conexões, fila) ou do
    horário do último deploy de um serviço. Serviços válidos: checkout-api,
    pagamentos-db, notificacoes-worker, auth-gateway.

    Args:
        servico: Nome exato do serviço, por exemplo "checkout-api".
    """
    metricas = _carregar_metricas()
    nome = servico.strip().lower()

    # ┌─ LACUNA [A1-L3] Erro vira observação (autocorreção do agente)
    # │  Dica: Não lance exceção! Devolva um TEXTO dizendo que o serviço não existe e liste os válidos — o agente vai ler e tentar de novo com o nome certo.
    # │  Copilot Chat: "Se o serviço não estiver em metricas, retorne uma string de erro amigável listando os serviços válidos separados por vírgula"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=1`
    if nome not in metricas:
        return f"ERRO: serviço '{servico}' não encontrado. Serviços válidos: {', '.join(sorted(metricas))}."

    return json.dumps({"servico": nome, **metricas[nome]}, ensure_ascii=False, indent=2)


@tool
def ler_logs(servico: str, filtro: str = "", max_linhas: int = 20) -> str:
    """Lê as linhas mais recentes do log de um serviço, com filtro opcional.

    Use para buscar EVIDÊNCIAS (mensagens de erro, timeouts, deploys). Prefira
    filtros como "ERROR", "WARN", "pool", "deploy" para economizar contexto.

    Args:
        servico: Nome exato do serviço, por exemplo "pagamentos-db".
        filtro: Texto que a linha precisa conter (sem diferenciar maiúsculas). Vazio = todas.
        max_linhas: Quantidade máxima de linhas retornadas (padrão 20, máximo 50).
    """
    arquivo = DIR_LOGS / f"{servico.strip().lower()}.log"
    if not arquivo.exists():
        disponiveis = sorted(p.stem for p in DIR_LOGS.glob("*.log"))
        return f"ERRO: não há log para '{servico}'. Logs disponíveis: {', '.join(disponiveis)}."

    linhas = arquivo.read_text(encoding="utf-8").splitlines()
    if filtro:
        linhas = [linha for linha in linhas if filtro.lower() in linha.lower()]
    max_linhas = max(1, min(int(max_linhas), 50))  # guarda-corpo de custo
    selecionadas = linhas[-max_linhas:]
    if not selecionadas:
        return f"Nenhuma linha em '{servico}' contém '{filtro}'."
    return "\n".join(selecionadas)


@tool
def buscar_runbook(consulta: str) -> str:
    """Busca nos runbooks internos o procedimento para um sintoma ou serviço.

    Use para saber COMO diagnosticar e mitigar um problema segundo o padrão da
    empresa (ex.: "latência alta checkout", "pool de conexões", "certificado").

    Args:
        consulta: Sintoma, serviço ou palavras-chave do problema.
    """
    termos = set(_normalizar(consulta))
    pontuados: list[tuple[int, str, str]] = []
    for arquivo in sorted(DIR_RUNBOOKS.glob("*.md")):
        texto = arquivo.read_text(encoding="utf-8")
        # Pontuação simples: quantos termos da consulta aparecem no runbook.
        # (Num projeto real, aqui entraria busca vetorial / RAG.)
        palavras = set(_normalizar(texto))
        pontos = sum(1 for termo in termos if termo in palavras or any(termo in p for p in palavras))
        if pontos:
            pontuados.append((pontos, arquivo.name, texto))

    if not pontuados:
        return "Nenhum runbook encontrado. Tente termos como: latência, pool, fila, certificado."

    pontuados.sort(key=lambda item: item[0], reverse=True)
    melhores = pontuados[:2]  # top-2 limita o contexto injetado
    return "\n\n---\n\n".join(f"[{nome}]\n{texto}" for _, nome, texto in melhores)


# --------------------------------------------------------------------------- #
# Ferramenta de ESCRITA (efeito colateral!)
# --------------------------------------------------------------------------- #


@tool
def abrir_ticket(
    titulo: str,
    servico: str,
    severidade: Literal["SEV1", "SEV2", "SEV3"],
    resumo: str,
) -> str:
    """Abre um ticket de incidente no sistema da empresa. TEM EFEITO COLATERAL.

    Só use depois de ter um diagnóstico com evidências. SEV1 = clientes sem
    conseguir comprar; SEV2 = degradação relevante; SEV3 = risco sem impacto atual.

    Args:
        titulo: Título curto e objetivo do incidente.
        servico: Serviço onde está a CAUSA RAIZ (não apenas o sintoma).
        severidade: SEV1, SEV2 ou SEV3.
        resumo: Causa raiz, evidências e ação recomendada, em poucas linhas.
    """
    DIR_TICKETS.mkdir(parents=True, exist_ok=True)
    agora = datetime.now(timezone.utc)
    ticket_id = f"OPS-{agora.strftime('%Y%m%d%H%M%S')}"
    ticket = {
        "id": ticket_id,
        "criado_em": agora.isoformat(),
        "titulo": titulo,
        "servico": servico,
        "severidade": severidade,
        "resumo": resumo,
        "origem": "opspilot",
    }
    (DIR_TICKETS / f"{ticket_id}.json").write_text(json.dumps(ticket, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Ticket {ticket_id} aberto ({severidade}) para {servico}."


# --------------------------------------------------------------------------- #
# Conjuntos de ferramentas reutilizados nas próximas aulas
# --------------------------------------------------------------------------- #

#: Só leitura: seguras para o agente usar livremente (e para sub-agentes).
FERRAMENTAS_LEITURA = [consultar_metricas, ler_logs, buscar_runbook]

#: Todas, incluindo a de escrita. Na Aula 2/3 ``abrir_ticket`` passa por aprovação humana.
TODAS_FERRAMENTAS = [*FERRAMENTAS_LEITURA, abrir_ticket]
