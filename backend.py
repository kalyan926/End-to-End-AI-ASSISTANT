from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler,BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool,Tool
from langchain_community.utilities import WikipediaAPIWrapper
import yfinance as yf
from langchain.agents import  AgentExecutor,create_react_agent
from typing import List, Union
import re
from langchain_core.prompts import PromptTemplate



app=FastAPI()


model_paths={"Llama3.2-1B":"./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
             "Llama3.2-3B":"./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
             "Qwen2.5-coder-1.5B":"./models/qwen2.5-coder-1.5b-instruct-q4_0.gguf"
             }


MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"  #set dynamically


#.....................RAG PART..............................
#nomic_model_here
embedding_model_path="./embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf"
nomic=LlamaCpp(model_path=embedding_model_path)
nomic_embeddings=LlamaCppEmbeddings(model_path=embedding_model_path)


model=""
reason=False
tools_selected=[]
vectorstore=""
doc_content=""


#.............................TOOLS..................................................

#tools for the AGENT

search=DuckDuckGoSearchRun()
search_tool=Tool(name="search",
                    func=search.run,
	            description="A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.")

@tool
def stock_price(ticker:str):
    '''This function is used to get latest price of stock using the ticker symbol.This function takes only ticker symbol as string'''
    out=yf.Ticker(ticker)
    price=out.history(period="1d")["Close"].values[0]
    return price


wikipedia = WikipediaAPIWrapper()
wikipedia_tool=Tool(name="Wikipedia",
                func=wikipedia.run,
	            description="A useful tool for searching the Internet to find information on stocks,world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")


@tool
def retriever_tool(query):
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 1,'score_threshold': 0.8,"lambda_mult":0.5,"fetch_k":20})
    # query translation here
    docs = retriever.get_relevant_documents(query)
    context=""
    for i in docs:
        context=context+i.page_content
    return context

#search_tool,executer_tool,wikipedia_tool,stock_price,retriever_tool

tools=[wikipedia_tool]

llama=""

def load_model():

    global llama

    llama = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=1,
        n_ctx=2048,
        max_tokens=512,
        top_k=1,
        streaming=True,
        verbose=False,
        use_cache=True )


class prompt(BaseModel):
    question:str

# Streaming generator for responses
async def response(prompt: prompt):
    
    if not reason and len(tools_selected)==0:
        """
        Generator function to stream LLaMA model responses.
        """
        for token in llama.stream(prompt):  # Streaming responses from the model
            yield token # Send each chunk of content

    elif not reason and tools_selected[0]=="retriever" and len(tools_selected)==1:

        """
        response from llama with retrieved context.
        """

        retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 1,'score_threshold': 0.8,"lambda_mult":0.5,"fetch_k":20})
        # query translation here
        docs = retriever.get_relevant_documents(prompt)
        
        rag_context=""
        for i in docs:
            rag_context=rag_context+i.page_content
        
        rag_prompt=prompt+rag_context
        for token in llama.stream(rag_prompt):  # Streaming responses from the model
            yield token  # Send each chunk of content

    elif reason:
        template = """Answer the following questions as best as you can. You have access to the following tools:
        {tools}
        Use the following format:
        Question: {input}
        {agent_scratchpad}"""

        react_prompt = PromptTemplate.from_template(template)

        tool_names = [tool.name for tool in tools_selected]
        agent = create_react_agent(llama,tools,react_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)
        return agent_executor.invoke({"input": prompt})      


class model_parameters(BaseModel):
    model:str
    tools:list
    reason:bool


@app.post("/set")
def set_model_parameters(model_p:model_parameters):

    global model,tools_selected,reason,MODEL_PATH
    MODEL_PATH=model_paths[model_p.model]
    model=model_p.model
    tools_selected=model_p.tools
    reason=model_p.reason
    load_model()
    return "sucessfully_set"


class documents(BaseModel):
      pdfs:list[UploadFile]=File(...)
      content:str

def context_length(text):
    return nomic.get_num_tokens(text)

@app.post("/store_docs")
def index_docs(files: List[UploadFile] = File(...)):

    text=""
    for file in files:
        pdf_reader=PdfReader(file.file)
        for page in pdf_reader.pages:
            text=text+page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(length_function=context_length,chunk_size=1300, chunk_overlap=0,separators=["\n\n", "\n", " ", ""])
    
    text_chunks = text_splitter.split_text(text)
    global vectorstore
    vectorstore = FAISS.from_texts(text_chunks, nomic_embeddings)

    return "succesfully created embeddings and stored"


@app.post("/stream")
async def stream_response(request: prompt):
    """
    Endpoint to stream model response.
    """
    
    if not request.question:
        return {"error": "No prompt provided"}
    
    # Create a streaming response using the generator
    return StreamingResponse(response(request.question), media_type="text/plain")
