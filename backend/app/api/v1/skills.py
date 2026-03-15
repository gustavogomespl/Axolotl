from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.project import Project
from app.models.skill import Skill

router = APIRouter(prefix="/projects/{project_id}/skills", tags=["skills"])


class SkillCreate(BaseModel):
    name: str
    description: str
    type: str  # "rag", "tool", "prompt"
    collection_name: str | None = None
    tool_names: list[str] | None = None
    system_prompt: str | None = None
    config: dict | None = None


class SkillUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    type: str | None = None
    collection_name: str | None = None
    tool_names: list[str] | None = None
    system_prompt: str | None = None
    config: dict | None = None
    is_active: bool | None = None


class SkillResponse(BaseModel):
    id: str
    project_id: str | None = None
    name: str
    description: str
    type: str
    collection_name: str | None = None
    tool_names: list[str] | None = None
    system_prompt: str | None = None
    is_active: bool = True
    config: dict | None = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


def _skill_to_response(skill: Skill) -> dict:
    return {
        "id": skill.id,
        "project_id": skill.project_id,
        "name": skill.name,
        "description": skill.description,
        "type": skill.type,
        "collection_name": skill.collection_name,
        "tool_names": skill.tool_names,
        "system_prompt": skill.system_prompt,
        "is_active": skill.is_active,
        "config": skill.config,
        "created_at": skill.created_at.isoformat() if skill.created_at else "",
        "updated_at": skill.updated_at.isoformat() if skill.updated_at else "",
    }


@router.post("")
async def create_skill(project_id: str, request: SkillCreate, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    collection_name = request.collection_name
    if request.type == "rag" and not collection_name:
        collection_name = f"skill_{request.name.lower().replace(' ', '_')}"

    skill = Skill(
        project_id=project_id,
        name=request.name,
        description=request.description,
        type=request.type,
        collection_name=collection_name,
        tool_names=request.tool_names,
        system_prompt=request.system_prompt,
        config=request.config,
    )
    db.add(skill)
    await db.commit()
    await db.refresh(skill)
    return _skill_to_response(skill)


@router.get("")
async def list_skills(project_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Skill).where(Skill.project_id == project_id).order_by(Skill.created_at.desc())
    )
    return [_skill_to_response(s) for s in result.scalars().all()]


@router.get("/{skill_id}")
async def get_skill(project_id: str, skill_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.project_id == project_id)
    )
    skill = result.scalars().first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_to_response(skill)


@router.put("/{skill_id}")
async def update_skill(
    project_id: str, skill_id: str, request: SkillUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.project_id == project_id)
    )
    skill = result.scalars().first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(skill, key, value)

    await db.commit()
    await db.refresh(skill)
    return _skill_to_response(skill)


@router.delete("/{skill_id}")
async def delete_skill(project_id: str, skill_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.project_id == project_id)
    )
    skill = result.scalars().first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    await db.delete(skill)
    await db.commit()
    return {"status": "deleted"}


@router.post("/{skill_id}/activate")
async def toggle_skill(project_id: str, skill_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Skill).where(Skill.id == skill_id, Skill.project_id == project_id)
    )
    skill = result.scalars().first()
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    skill.is_active = not skill.is_active
    await db.commit()
    return {"is_active": skill.is_active}
