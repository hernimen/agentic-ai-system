from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(description="The message to be analyzed")
    thread_id: str = Field(description="The ID of the thread")


class ChatResponse(BaseModel):
    response: str
    thread_id: str
