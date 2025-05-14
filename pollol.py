import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse  # Added missing import
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # Fixed import
from dotenv import load_dotenv

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["LANGCHAIN_COMMUNITY_NO_PEBBLO"] = "1"

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

app = FastAPI()

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

# PDF Agent implementation
class PDFAgent:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200  # Fixed missing closing parenthesis
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

    async def process_pdf(self, file_path: str):
        """Process PDF using LangChain components"""
        try:
            # Load and split PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            # Initialize QA chain
            self._init_qa_chain()
            
            return {"message": "PDF processed successfully", "chunks": len(chunks)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    def _init_qa_chain(self):
        """Initialize the QA chain using LCEL"""
        template = """Use the following context to answer the question in a detailed manner. 
        If you don't know the answer, mention that the provided pdf does not answer the question and then provide the correct answer based on your expertise. Be detailed and helpful.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Fixed LCEL chain syntax
        self.qa_chain = (
            RunnableParallel({
                "context": self.retriever,
                "question": RunnablePassthrough()
            })
            | prompt
            | ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.7,
                api_key=os.getenv("GROQ_API_KEY")  # Get API key from environment
            )
            | StrOutputParser()
        )

    async def ask_question(self, question: str) -> str:
        if not self.qa_chain:
            raise HTTPException(status_code=400, detail="No document processed yet")
            
        try:
            return await self.qa_chain.ainvoke(question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

pdf_agent = PDFAgent()

# FastAPI endpoints
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_file = Path("temp.pdf")
    try:
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        await pdf_agent.process_pdf(str(temp_file))
        return {"message": "PDF uploaded and processed!"}
    finally:
        if temp_file.exists():
            temp_file.unlink()

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    return {"answer": await pdf_agent.ask_question(request.question)}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()
