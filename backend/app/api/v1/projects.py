from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.project import Project

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    planner_prompt: str | None = None
    model: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    planner_prompt: str | None = None
    model: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    planner_prompt: str | None = None
    model: str | None = None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


@router.post("", response_model=ProjectResponse)
async def create_project(request: ProjectCreate, db: AsyncSession = Depends(get_db)):
    project = Project(
        name=request.name,
        description=request.description,
        planner_prompt=request.planner_prompt,
        model=request.model,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("")
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    return [ProjectResponse.model_validate(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str, request: ProjectUpdate, db: AsyncSession = Depends(get_db)
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(project, key, value)

    await db.commit()
    await db.refresh(project)
    return project


@router.delete("/{project_id}")
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await db.delete(project)
    await db.commit()
    return {"status": "deleted"}
