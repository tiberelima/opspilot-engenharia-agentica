# Aula 3 · Deep Agent Harness

**Resultado:** um "Comandante de Incidente" que planeja, delega em paralelo, escreve arquivos e aprende.

| Pilar | Onde | Lacunas |
|---|---|---|
| Sub-agentes | `opspilot/deep_agent.py` | A3-L1, A3-L2 (o grafo da Aula 2 vira sub-agente) |
| Sistema de arquivos | `opspilot/deep_agent.py` | A3-L3 |
| Planejamento + Memória + tudo ligado | `opspilot/deep_agent.py` | A3-L4 |
| Humano no loop do harness | `a3_01_orquestrador.py` | A3-L5 |

```bash
make bonde AULA=3        # (só se ficou para trás)
make test AULA=3
make aula3
git diff workspace/memoria   # o que o agente aprendeu?
```

**Checkpoint:** `workspace/relatorios/postmortem-INC-2041.md` existe, segue o formato do `AGENTS.md`
e aponta **pagamentos-db** (pool 50→10) como causa raiz, não o deploy do checkout.
