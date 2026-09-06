from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(description="The message to be analyzed")
    thread_id: str = Field(description="The ID of the thread")


class Citation(BaseModel):
    source_name: str
    document_id: str
    title: str
    section: str
    pages: list[int] = Field(default_factory=list)
    excerpt: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool | None = None
