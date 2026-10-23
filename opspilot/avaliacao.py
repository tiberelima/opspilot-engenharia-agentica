"""Aula 4 — Avaliação automática de agentes (o "teste unitário" da IA).

Três camadas, da mais barata para a mais cara
---------------------------------------------
1. **Regras determinísticas** (grátis, instantâneas): a categoria está certa?
   o serviço da causa raiz é o esperado? a resposta cita os termos-chave?
   contém algo proibido? → :func:`avaliar_regras`.
2. **LLM-as-a-judge** (centavos): um segundo modelo dá nota de 1 a 5 para a
   *fundamentação* — cada afirmação está apoiada no que as ferramentas
   retornaram? É a principal arma contra alucinação. → :func:`julgar_fundamentacao`.
3. **Humano** (caro): amostragem no painel do Langfuse (anotações manuais).

O resultado vira (a) scores no Langfuse e (b) um **gate de CI**: se a média
cair abaixo do limite, o pipeline falha e o PR não entra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from opspilot.config import DIR_DADOS, DIR_WORKSPACE, MODELO_JUIZ, obter_llm

ARQ_CASOS = DIR_DADOS / "avaliacao" / "casos.jsonl"


@dataclass
class Caso:
    """Um caso do *golden dataset* (``data/avaliacao/casos.jsonl``)."""

    id: str
    pergunta: str
    categoria_esperada: str
    servico_causa_esperado: str | None = None
    termos_causa: list[str] = field(default_factory=list)
    termos_proibidos: list[str] = field(default_factory=list)


def carregar_casos(caminho: Path = ARQ_CASOS) -> list[Caso]:
    """Lê o dataset JSONL (um caso por linha)."""
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    return [Caso(**json.loads(linha)) for linha in linhas if linha.strip()]


# --------------------------------------------------------------------------- #
# Camada 1 — regras determinísticas
# --------------------------------------------------------------------------- #


def avaliar_regras(caso: Caso, resposta: str, estado: dict) -> dict[str, float]:
    """Aplica checagens objetivas. Cada critério vale 0.0 ou 1.0.

    Critérios que não se aplicam ao caso (ex.: serviço numa pergunta fora de
    escopo) são omitidos — não inflam nem derrubam a média.
    """
    texto = resposta.lower()
    notas: dict[str, float] = {
        "categoria_correta": float(estado.get("categoria") == caso.categoria_esperada),
        "sem_termos_proibidos": float(not any(t.lower() in texto for t in caso.termos_proibidos)),
    }
    if caso.termos_causa:
        notas["cita_termos_causa"] = float(all(t.lower() in texto for t in caso.termos_causa))
    if caso.servico_causa_esperado:
        diag = estado.get("diagnostico") or {}
        # ┌─ LACUNA [A4-L3] Regra: a causa raiz está no serviço certo?
        # │  Dica: Compare diag.get("servico_causa_raiz") com caso.servico_causa_esperado e converta o booleano para float.
        # │  Copilot Chat: "Atribua a notas["servico_correto"] o float da comparação entre diag.get("servico_causa_raiz") e caso.servico_causa_esperado"
        # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=4`
        notas["servico_correto"] = float(diag.get("servico_causa_raiz") == caso.servico_causa_esperado)
    return notas


# --------------------------------------------------------------------------- #
# Camada 2 — LLM-as-a-judge
# --------------------------------------------------------------------------- #


class NotaJuiz(BaseModel):
    """Saída estruturada do juiz."""

    nota: int = Field(ge=1, le=5, description="1 = inventou fatos; 5 = tudo apoiado em evidência.")
    justificativa: str = Field(description="Uma ou duas frases citando o principal problema ou acerto.")


PROMPT_JUIZ = """Você é um auditor de qualidade de respostas de um agente de SRE.
Avalie a FUNDAMENTAÇÃO da RESPOSTA: cada número, horário, serviço e conclusão
precisa estar apoiado nas EVIDÊNCIAS DAS FERRAMENTAS. Penalize forte qualquer
fato inventado. Recusas corretas de perguntas fora de escopo valem 5.

PERGUNTA:
{pergunta}

EVIDÊNCIAS DAS FERRAMENTAS:
{evidencias}

RESPOSTA DO AGENTE:
{resposta}
"""


def julgar_fundamentacao(pergunta: str, resposta: str, evidencias: str) -> NotaJuiz:
    """Pede a um modelo-juiz uma nota de 1 a 5 para a fundamentação da resposta."""
    # ┌─ LACUNA [A4-L4] Monte o juiz com saída estruturada
    # │  Dica: Igual ao classificador da Aula 2: obter_llm(MODELO_JUIZ).with_structured_output(NotaJuiz) e depois .invoke(prompt formatado). Limite as evidências a ~6000 caracteres para controlar custo.
    # │  Copilot Chat: "Crie juiz com obter_llm(MODELO_JUIZ).with_structured_output(NotaJuiz) e retorne juiz.invoke(PROMPT_JUIZ.format(...)) com evidencias truncadas em 6000 caracteres"
    # └─ Troque cada ___ abaixo. Gabarito: `make gabarito-aula AULA=4`
    juiz = obter_llm(MODELO_JUIZ).with_structured_output(NotaJuiz)
    return juiz.invoke(
        PROMPT_JUIZ.format(pergunta=pergunta, resposta=resposta, evidencias=evidencias[:6000] or "(nenhuma)")
    )


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #


def salvar_relatorio(resultados: list[dict], limite: float) -> tuple[float, Path]:
    """Calcula a nota geral, grava JSON + Markdown em ``workspace/relatorios``.

    Returns:
        ``(nota_geral, caminho_do_markdown)``. Nota geral = média das notas de cada caso (0 a 1).
    """
    geral = mean(r["nota_final"] for r in resultados) if resultados else 0.0
    destino = DIR_WORKSPACE / "relatorios"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "avaliacao.json").write_text(
        json.dumps({"nota_geral": geral, "limite": limite, "casos": resultados}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    linhas = [
        "# Relatório de avaliação do OpsPilot",
        "",
        f"**Nota geral:** {geral:.2f} (limite do gate: {limite:.2f}) → {'✅ APROVADO' if geral >= limite else '❌ REPROVADO'}",
        "",
        "| Caso | Regras | Juiz (1-5) | Nota | Trace |",
        "|---|---|---|---|---|",
    ]
    for r in resultados:
        trace = f"[abrir]({r['trace_url']})" if r.get("trace_url") else "—"
        linhas.append(
            f"| {r['id']} | {r['nota_regras']:.2f} | {r.get('nota_juiz', '—')} | {r['nota_final']:.2f} | {trace} |"
        )
    md = destino / "avaliacao.md"
    md.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return geral, md
