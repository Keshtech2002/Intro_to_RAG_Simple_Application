#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from rag import load_db
from langserve import add_routes

rag_chain = load_db("Relunctant_to_go_but_Victory_came_back.txt", k=3)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Add the rag_chain with the correct path
#app.include_router(rag_chain, prefix="/rag_chain")

add_routes(
    app,
    rag_chain,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)