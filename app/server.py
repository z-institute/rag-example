from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import asyncio

load_dotenv()

from app.rag_chain import final_chain


# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="A simple api server using Langchain's Runnable interfaces",
# )

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai",
# )

# add_routes(
#     app,
#     ChatAnthropic(),
#     path="/anthropic",
# )

# model = ChatAnthropic()
# prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
# add_routes(
#     app,
#     prompt | model,
#     path="/joke",
# )
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, final_chain, path="/rag", playground_type="default")

@app.get("/sse")
async def sse_endpoint(request: Request):
    async def event_generator():
        try:
            while True:
                # Your SSE logic here
                yield {"data": "Some data"}
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("Client disconnected")

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
