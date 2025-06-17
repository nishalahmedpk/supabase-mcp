from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

app = FastAPI()

model = AzureChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-35-turbo",
    api_version="2025-01-01-preview",
)

server_params = StdioServerParameters(
    command="npx",
    env={
        "SUPABASE_ACCESS_TOKEN": os.getenv("SUPABASE"),
    },
    args=["-y", "@supabase/mcp-server-supabase@latest"]
)

class ChatRequest(BaseModel):
    messages: list

@app.on_event("startup")
async def startup_event():
    app.state.tools = None
    app.state.agent = None
    app.state.session = None
    app.state.stdio_ctx = None

    app.state.stdio_ctx = stdio_client(server_params)
    read, write = await app.state.stdio_ctx.__aenter__()
    app.state.session = ClientSession(read, write)
    await app.state.session.__aenter__()
    await app.state.session.initialize()
    tools = await load_mcp_tools(app.state.session)
    MAX_DESCRIPTION_LENGTH = 200
    for tool in tools:
        if hasattr(tool, "description") and isinstance(tool.description, str):
            if len(tool.description) > MAX_DESCRIPTION_LENGTH:
                tool.description = tool.description[:MAX_DESCRIPTION_LENGTH] + "..."
    app.state.tools = tools
    app.state.agent = create_react_agent(model, tools)

@app.on_event("shutdown")
async def shutdown_event():
    if app.state.session:
        await app.state.session.__aexit__(None, None, None)
    if app.state.stdio_ctx:
        await app.state.stdio_ctx.__aexit__(None, None, None)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        agent = app.state.agent
        messages = request.messages
        agent_response = await agent.ainvoke({"messages": messages})
        ai_message = agent_response["messages"][-1].content
        return JSONResponse(content={"response": ai_message})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)