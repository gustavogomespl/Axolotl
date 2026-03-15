import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/skills", tags=["skills"])

# In-memory store (will be replaced by SQLAlchemy + PostgreSQL)
_skills: dict[str, dict] = {}


class SkillCreate(BaseModel):
    name: str
    description: str
    type: str  # "rag", "tool", "subgraph", "prompt"
    collection_name: str | None = None
    tool_names: list[str] | None = None
    subgraph_name: str | None = None
    system_prompt: str | None = None
    config: dict | None = None


class SkillUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    type: str | None = None
    collection_name: str | None = None
    tool_names: list[str] | None = None
    subgraph_name: str | None = None
    system_prompt: str | None = None
    config: dict | None = None
    is_active: bool | None = None


class SkillResponse(BaseModel):
    id: str
    name: str
    description: str
    type: str
    collection_name: str | None = None
    tool_names: list[str] | None = None
    subgraph_name: str | None = None
    system_prompt: str | None = None
    is_active: bool = True
    config: dict | None = None
    created_at: str
    updated_at: str


@router.post("", response_model=SkillResponse)
async def create_skill(request: SkillCreate):
    """Create a new skill."""
    skill_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # For RAG skills, auto-create collection name if not provided
    collection_name = request.collection_name
    if request.type == "rag" and not collection_name:
        collection_name = f"skill_{request.name.lower().replace(' ', '_')}"

    skill = {
        "id": skill_id,
        "name": request.name,
        "description": request.description,
        "type": request.type,
        "collection_name": collection_name,
        "tool_names": request.tool_names,
        "subgraph_name": request.subgraph_name,
        "system_prompt": request.system_prompt,
        "is_active": True,
        "config": request.config,
        "created_at": now,
        "updated_at": now,
    }
    _skills[skill_id] = skill
    return SkillResponse(**skill)


@router.get("")
async def list_skills():
    """List all skills."""
    return list(_skills.values())


@router.get("/{skill_id}", response_model=SkillResponse)
async def get_skill(skill_id: str):
    """Get skill details."""
    skill = _skills.get(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return SkillResponse(**skill)


@router.put("/{skill_id}", response_model=SkillResponse)
async def update_skill(skill_id: str, request: SkillUpdate):
    """Update a skill."""
    skill = _skills.get(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    update_data = request.model_dump(exclude_unset=True)
    skill.update(update_data)
    skill["updated_at"] = datetime.utcnow().isoformat()
    return SkillResponse(**skill)


@router.delete("/{skill_id}")
async def delete_skill(skill_id: str):
    """Delete a skill."""
    if skill_id not in _skills:
        raise HTTPException(status_code=404, detail="Skill not found")
    del _skills[skill_id]
    return {"status": "deleted"}


@router.post("/{skill_id}/activate")
async def toggle_skill(skill_id: str):
    """Toggle skill active state."""
    skill = _skills.get(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    skill["is_active"] = not skill["is_active"]
    return {"is_active": skill["is_active"]}
