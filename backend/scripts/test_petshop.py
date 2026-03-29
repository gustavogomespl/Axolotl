"""Roteiro de testes para validar a integracao DeepAgents + PetShop.

Executa testes automatizados contra a API do Axolotl para verificar:
1. Saude do backend
2. CRUD de projeto com orchestration_mode
3. CRUD de skills, tools, agentes
4. Chat sincrono com deep_agent
5. Chat streaming com deep_agent
6. Cenarios de atendimento do petshop

Usage:
  # Via Docker:
  docker compose exec backend python scripts/test_petshop.py

  # Local:
  uv run python scripts/test_petshop.py

  # Pular testes de chat (nao precisa de LLM key):
  SKIP_CHAT=1 python scripts/test_petshop.py
"""

import json
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("AXOLOTL_API_URL", "http://localhost:8000/api/v1")
SKIP_CHAT = os.environ.get("SKIP_CHAT", "").lower() in ("1", "true", "yes")

passed = 0
failed = 0
skipped = 0


def test(name: str, func):
    """Run a test and report result."""
    global passed, failed, skipped
    try:
        result = func()
        if result == "SKIP":
            print(f"  [SKIP] {name}")
            skipped += 1
        else:
            print(f"  [PASS] {name}")
            passed += 1
        return result
    except Exception as e:
        print(f"  [FAIL] {name}")
        print(f"         {e}")
        failed += 1
        return None


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: esperado {expected!r}, recebeu {actual!r}")


def assert_in(value, container, msg=""):
    if value not in container:
        raise AssertionError(f"{msg}: {value!r} nao encontrado em {container!r}")


def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(msg or "Condicao falsa")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
client = httpx.Client(base_url=BASE_URL, timeout=30)
project_id = None
skill_ids = {}
tool_ids = {}
agent_ids = {}


# ===========================================================================
print("\n" + "=" * 60)
print("  ROTEIRO DE TESTES - Axolotl + DeepAgents + PetShop")
print("=" * 60)


# ===========================================================================
# FASE 1: Saude do Backend
# ===========================================================================
print("\n--- FASE 1: Saude do Backend ---")


def test_health():
    health_url = BASE_URL + "/health"
    resp = httpx.get(health_url, timeout=10)
    assert_eq(resp.status_code, 200, "Health check")
    data = resp.json()
    assert_in(data["status"], ("healthy", "ok"), "Status")


test("GET /health responde 200", test_health)


def test_api_root():
    resp = client.get("/projects")
    assert_eq(resp.status_code, 200, "Lista de projetos")
    assert_true(isinstance(resp.json(), list), "Resposta e uma lista")


test("GET /projects responde 200", test_api_root)


# ===========================================================================
# FASE 2: Projeto com orchestration_mode
# ===========================================================================
print("\n--- FASE 2: Projeto com orchestration_mode ---")


def test_create_project_deep_agent():
    global project_id
    resp = client.post("/projects", json={
        "name": f"Test PetShop {int(time.time())}",
        "description": "Projeto de teste automatizado",
        "orchestration_mode": "deep_agent",
        "model": "openai:gpt-4.1-mini",
        "planner_prompt": "Voce e uma recepcionista de petshop.",
    })
    assert_eq(resp.status_code, 200, "Criar projeto")
    data = resp.json()
    assert_eq(data["orchestration_mode"], "deep_agent", "Modo deep_agent")
    assert_true(data["id"], "Project ID existe")
    project_id = data["id"]
    return project_id


test("Criar projeto com orchestration_mode=deep_agent", test_create_project_deep_agent)


def test_get_project_has_mode():
    resp = client.get(f"/projects/{project_id}")
    assert_eq(resp.status_code, 200, "GET projeto")
    data = resp.json()
    assert_eq(data["orchestration_mode"], "deep_agent", "Modo persiste")


test("GET projeto retorna orchestration_mode", test_get_project_has_mode)


def test_update_project_mode():
    resp = client.put(f"/projects/{project_id}", json={"orchestration_mode": "supervisor"})
    assert_eq(resp.status_code, 200, "Update modo")
    assert_eq(resp.json()["orchestration_mode"], "supervisor", "Modo atualizado")
    # Voltar para deep_agent
    client.put(f"/projects/{project_id}", json={"orchestration_mode": "deep_agent"})


test("Atualizar orchestration_mode via PUT", test_update_project_mode)


def test_invalid_mode():
    resp = client.post("/projects", json={
        "name": f"Invalid {int(time.time())}",
        "orchestration_mode": "modo_invalido",
    })
    assert_eq(resp.status_code, 422, "Modo invalido retorna 422")


test("Modo invalido retorna 422", test_invalid_mode)


def test_create_project_simple():
    resp = client.post("/projects", json={
        "name": f"Simple {int(time.time())}",
        "orchestration_mode": "simple",
        "model": "openai:gpt-4.1-mini",
    })
    assert_eq(resp.status_code, 200, "Criar projeto simple")
    assert_eq(resp.json()["orchestration_mode"], "simple", "Modo simple")
    # Limpar
    client.delete(f"/projects/{resp.json()['id']}")


test("Criar projeto com orchestration_mode=simple", test_create_project_simple)


def test_default_mode_is_supervisor():
    resp = client.post("/projects", json={
        "name": f"Default {int(time.time())}",
        "model": "openai:gpt-4.1-mini",
    })
    assert_eq(resp.status_code, 200, "Criar projeto default")
    assert_eq(resp.json()["orchestration_mode"], "supervisor", "Default e supervisor")
    client.delete(f"/projects/{resp.json()['id']}")


test("Modo padrao e 'supervisor'", test_default_mode_is_supervisor)


# ===========================================================================
# FASE 3: Skills (Base de Conhecimento)
# ===========================================================================
print("\n--- FASE 3: Skills ---")


def test_create_skill_rag():
    resp = client.post(f"/projects/{project_id}/skills", json={
        "name": "tabela-precos-test",
        "description": "Tabela de precos para teste",
        "type": "rag",
        "system_prompt": "Use para consultar precos.",
    })
    assert_eq(resp.status_code, 200, "Criar skill RAG")
    data = resp.json()
    assert_eq(data["type"], "rag", "Tipo RAG")
    assert_true(data["is_active"], "Skill ativa por padrao")
    assert_true(data["collection_name"], "Collection name gerado")
    skill_ids["tabela-precos"] = data["id"]


test("Criar skill tipo RAG", test_create_skill_rag)


def test_create_skill_prompt():
    resp = client.post(f"/projects/{project_id}/skills", json={
        "name": "cuidados-saude-test",
        "description": "Orientacoes de saude",
        "type": "rag",
        "system_prompt": "Oriente sobre cuidados basicos.",
    })
    assert_eq(resp.status_code, 200, "Criar skill")
    skill_ids["cuidados-saude"] = resp.json()["id"]


test("Criar skill de cuidados/saude", test_create_skill_prompt)


def test_list_skills():
    resp = client.get(f"/projects/{project_id}/skills")
    assert_eq(resp.status_code, 200, "Listar skills")
    assert_true(len(resp.json()) >= 2, f"Pelo menos 2 skills ({len(resp.json())} encontradas)")


test("Listar skills do projeto", test_list_skills)


def test_toggle_skill():
    sid = skill_ids["tabela-precos"]
    resp = client.post(f"/projects/{project_id}/skills/{sid}/activate")
    assert_eq(resp.status_code, 200, "Toggle skill")
    assert_eq(resp.json()["is_active"], False, "Desativada")
    # Reativar
    client.post(f"/projects/{project_id}/skills/{sid}/activate")


test("Toggle ativar/desativar skill", test_toggle_skill)


# ===========================================================================
# FASE 4: Tools (APIs externas)
# ===========================================================================
print("\n--- FASE 4: Tools ---")


def test_create_tool_api():
    resp = client.post(f"/projects/{project_id}/tools", json={
        "name": "consultar-agenda-test",
        "description": "Consulta horarios disponiveis",
        "type": "api",
        "category": "agendamento",
        "api_config": {
            "name": "consultar-agenda-test",
            "description": "Retorna horarios disponiveis",
            "method": "GET",
            "url": "https://api.petshop.com/agenda/disponibilidade",
            "headers": {},
            "auth_type": "none",
            "auth_config": {},
        },
    })
    assert_eq(resp.status_code, 200, "Criar tool API")
    data = resp.json()
    assert_eq(data["type"], "api", "Tipo API")
    assert_true(data["api_config"], "api_config presente")
    tool_ids["consultar-agenda"] = data["id"]


test("Criar tool tipo API", test_create_tool_api)


def test_create_tool_estoque():
    resp = client.post(f"/projects/{project_id}/tools", json={
        "name": "consultar-estoque-test",
        "description": "Consulta estoque de produtos",
        "type": "api",
        "category": "produtos",
        "api_config": {
            "name": "consultar-estoque-test",
            "description": "Consulta estoque",
            "method": "GET",
            "url": "https://api.petshop.com/estoque/consulta",
            "headers": {},
            "auth_type": "none",
            "auth_config": {},
        },
    })
    assert_eq(resp.status_code, 200, "Criar tool estoque")
    tool_ids["consultar-estoque"] = resp.json()["id"]


test("Criar tool de estoque", test_create_tool_estoque)


def test_list_tools():
    resp = client.get(f"/projects/{project_id}/tools")
    assert_eq(resp.status_code, 200, "Listar tools")
    assert_true(len(resp.json()) >= 2, f"Pelo menos 2 tools ({len(resp.json())} encontradas)")


test("Listar tools do projeto", test_list_tools)


# ===========================================================================
# FASE 5: Agentes (Planner + Workers)
# ===========================================================================
print("\n--- FASE 5: Agentes ---")


def test_create_planner():
    resp = client.post(f"/projects/{project_id}/agents", json={
        "name": "recepcionista-test",
        "description": "Recepcionista virtual de teste",
        "is_planner": True,
        "model": "openai:gpt-4.1-mini",
        "prompt": "Voce e a recepcionista do PetShop. Delegue para sub-agentes.",
        "skill_ids": list(skill_ids.values()),
    })
    assert_eq(resp.status_code, 200, "Criar planner")
    data = resp.json()
    assert_eq(data["is_planner"], True, "is_planner=True")
    assert_true(len(data["skill_ids"]) >= 2, "Skills linkadas")
    agent_ids["planner"] = data["id"]


test("Criar agente planner (recepcionista)", test_create_planner)


def test_create_worker_agendamento():
    resp = client.post(f"/projects/{project_id}/agents", json={
        "name": "agendamento-agent-test",
        "description": "Especialista em agendamentos de banho, tosa e hospedagem",
        "is_planner": False,
        "model": "openai:gpt-4.1-mini",
        "prompt": "Voce e o especialista em agendamentos do PetShop.",
        "tool_ids": [tool_ids["consultar-agenda"]],
        "skill_ids": [skill_ids["tabela-precos"]],
    })
    assert_eq(resp.status_code, 200, "Criar worker agendamento")
    data = resp.json()
    assert_eq(data["is_planner"], False, "is_planner=False")
    assert_true(len(data["tool_ids"]) >= 1, "Tools linkadas")
    assert_true(len(data["skill_ids"]) >= 1, "Skills linkadas")
    agent_ids["agendamento"] = data["id"]


test("Criar worker: agendamento-agent", test_create_worker_agendamento)


def test_create_worker_produtos():
    resp = client.post(f"/projects/{project_id}/agents", json={
        "name": "consultor-produtos-test",
        "description": "Especialista em racoes e acessorios para pets",
        "is_planner": False,
        "prompt": "Voce e o consultor de produtos do PetShop.",
        "tool_ids": [tool_ids["consultar-estoque"]],
    })
    assert_eq(resp.status_code, 200, "Criar worker produtos")
    agent_ids["produtos"] = resp.json()["id"]


test("Criar worker: consultor-produtos", test_create_worker_produtos)


def test_create_worker_saude():
    resp = client.post(f"/projects/{project_id}/agents", json={
        "name": "saude-pet-test",
        "description": "Orientador sobre cuidados e saude pet. NAO faz diagnosticos.",
        "is_planner": False,
        "prompt": "Voce e o orientador de saude pet. NUNCA faca diagnosticos.",
        "skill_ids": [skill_ids["cuidados-saude"]],
    })
    assert_eq(resp.status_code, 200, "Criar worker saude")
    agent_ids["saude"] = resp.json()["id"]


test("Criar worker: saude-pet", test_create_worker_saude)


def test_list_agents():
    resp = client.get(f"/projects/{project_id}/agents")
    assert_eq(resp.status_code, 200, "Listar agentes")
    agents = resp.json()
    assert_eq(len(agents), 4, "4 agentes criados (1 planner + 3 workers)")
    planners = [a for a in agents if a["is_planner"]]
    workers = [a for a in agents if not a["is_planner"]]
    assert_eq(len(planners), 1, "Exatamente 1 planner")
    assert_eq(len(workers), 3, "Exatamente 3 workers")


test("Listar agentes: 1 planner + 3 workers", test_list_agents)


def test_agent_has_linked_resources():
    resp = client.get(f"/projects/{project_id}/agents/{agent_ids['agendamento']}")
    assert_eq(resp.status_code, 200, "GET agente")
    data = resp.json()
    assert_in(tool_ids["consultar-agenda"], data["tool_ids"], "Tool linkada")
    assert_in(skill_ids["tabela-precos"], data["skill_ids"], "Skill linkada")


test("Agente tem tools e skills linkados", test_agent_has_linked_resources)


# ===========================================================================
# FASE 6: Chat com Deep Agent
# ===========================================================================
print("\n--- FASE 6: Chat com Deep Agent ---")


def test_chat_sync():
    if SKIP_CHAT:
        return "SKIP"
    resp = client.post(
        f"/projects/{project_id}/chat",
        json={"message": "Oi, tudo bem?"},
        timeout=60,
    )
    assert_eq(resp.status_code, 200, "Chat sincrono")
    data = resp.json()
    assert_true(data["thread_id"], "thread_id presente")
    assert_true(data["response"], "Resposta nao vazia")
    assert_true(len(data["response"]) > 10, f"Resposta substancial ({len(data['response'])} chars)")
    return data


test("Chat sincrono com deep_agent", test_chat_sync)


def test_chat_stream():
    if SKIP_CHAT:
        return "SKIP"
    resp = client.post(
        f"/projects/{project_id}/chat/stream",
        json={"message": "Quanto custa um banho?"},
        timeout=60,
    )
    assert_eq(resp.status_code, 200, "Chat streaming")
    assert_in("text/event-stream", resp.headers.get("content-type", ""), "Content-Type SSE")

    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                event = json.loads(line[6:])
                events.append(event)
            except json.JSONDecodeError:
                pass

    event_types = [e["type"] for e in events]
    assert_in("metadata", event_types, "Evento metadata")
    assert_in("done", event_types, "Evento done")
    # Deve ter pelo menos metadata + tokens + done
    assert_true(len(events) >= 3, f"Pelo menos 3 eventos SSE ({len(events)} recebidos)")
    # Verificar que metadata tem thread_id
    metadata = next(e for e in events if e["type"] == "metadata")
    assert_true(metadata.get("thread_id"), "thread_id no metadata")


test("Chat streaming com SSE events", test_chat_stream)


def test_chat_agendamento():
    if SKIP_CHAT:
        return "SKIP"
    resp = client.post(
        f"/projects/{project_id}/chat",
        json={"message": "Quero agendar um banho pro meu cachorro Rex, ele e um labrador grande"},
        timeout=60,
    )
    assert_eq(resp.status_code, 200, "Chat agendamento")
    response_text = resp.json()["response"].lower()
    # O agente deve responder sobre agendamento (nao importa o conteudo exato)
    assert_true(len(response_text) > 20, "Resposta substancial sobre agendamento")


test("Cenario: Agendamento de banho", test_chat_agendamento)


def test_chat_emergencia():
    if SKIP_CHAT:
        return "SKIP"
    resp = client.post(
        f"/projects/{project_id}/chat",
        json={"message": "Meu gato esta vomitando sangue, o que eu faco?"},
        timeout=60,
    )
    assert_eq(resp.status_code, 200, "Chat emergencia")
    response_text = resp.json()["response"].lower()
    # Deve mencionar veterinario ou emergencia
    has_urgency = any(w in response_text for w in ["veterinari", "emergencia", "urgente", "imediatamente", "clinica", "98888"])
    assert_true(has_urgency, "Resposta menciona veterinario/emergencia")


test("Cenario: Emergencia veterinaria", test_chat_emergencia)


def test_chat_thread_persistence():
    if SKIP_CHAT:
        return "SKIP"
    # Primeira mensagem
    resp1 = client.post(
        f"/projects/{project_id}/chat",
        json={"message": "Meu nome e Joao e tenho um poodle chamado Bolt"},
        timeout=60,
    )
    assert_eq(resp1.status_code, 200, "Chat 1")
    thread_id = resp1.json()["thread_id"]

    # Segunda mensagem no mesmo thread
    resp2 = client.post(
        f"/projects/{project_id}/chat",
        json={"message": "Qual servico voce recomenda pro meu cachorro?", "thread_id": thread_id},
        timeout=60,
    )
    assert_eq(resp2.status_code, 200, "Chat 2 no mesmo thread")
    assert_eq(resp2.json()["thread_id"], thread_id, "Mesmo thread_id")


test("Persistencia de thread (multi-turno)", test_chat_thread_persistence)


# ===========================================================================
# FASE 7: Fallback e Edge Cases
# ===========================================================================
print("\n--- FASE 7: Fallback e Edge Cases ---")


def test_simple_mode_fallback():
    if SKIP_CHAT:
        return "SKIP"
    # Criar projeto simple
    resp = client.post("/projects", json={
        "name": f"Simple Fallback {int(time.time())}",
        "orchestration_mode": "simple",
        "model": "openai:gpt-4.1-mini",
        "planner_prompt": "Responda de forma breve.",
    })
    assert_eq(resp.status_code, 200, "Criar projeto simple")
    simple_id = resp.json()["id"]

    # Chat sem agentes - deve usar simple agent
    resp2 = client.post(
        f"/projects/{simple_id}/chat",
        json={"message": "Ola"},
        timeout=60,
    )
    assert_eq(resp2.status_code, 200, "Chat no modo simple")
    assert_true(resp2.json()["response"], "Resposta nao vazia")

    # Limpar
    client.delete(f"/projects/{simple_id}")


test("Modo simple funciona sem agentes", test_simple_mode_fallback)


def test_deep_agent_no_workers_fallback():
    if SKIP_CHAT:
        return "SKIP"
    # Criar projeto deep_agent SEM workers
    resp = client.post("/projects", json={
        "name": f"No Workers {int(time.time())}",
        "orchestration_mode": "deep_agent",
        "model": "openai:gpt-4.1-mini",
    })
    simple_id = resp.json()["id"]

    # Deve fazer fallback para simple agent
    resp2 = client.post(
        f"/projects/{simple_id}/chat",
        json={"message": "Ola"},
        timeout=60,
    )
    assert_eq(resp2.status_code, 200, "Fallback para simple")
    assert_true(resp2.json()["response"], "Resposta no fallback")

    client.delete(f"/projects/{simple_id}")


test("Deep agent sem workers faz fallback para simple", test_deep_agent_no_workers_fallback)


# ===========================================================================
# FASE 8: Limpeza
# ===========================================================================
print("\n--- FASE 8: Limpeza ---")


def test_cleanup():
    if project_id:
        resp = client.delete(f"/projects/{project_id}")
        assert_eq(resp.status_code, 200, "Deletar projeto de teste")
        assert_eq(resp.json()["status"], "deleted", "Status deleted")


test("Deletar projeto de teste (cascade)", test_cleanup)


# ===========================================================================
# Resultado Final
# ===========================================================================
print("\n" + "=" * 60)
total = passed + failed + skipped
print(f"  RESULTADO: {passed} passed, {failed} failed, {skipped} skipped ({total} total)")
print("=" * 60)

if SKIP_CHAT:
    print("\n  NOTA: Testes de chat pulados (SKIP_CHAT=1)")
    print("  Para rodar completo: python scripts/test_petshop.py")

if failed:
    print(f"\n  {failed} teste(s) falharam!")
    sys.exit(1)
else:
    print("\n  Todos os testes passaram!")
    sys.exit(0)
