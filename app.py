import os
import logging
from typing import Dict, Any, List, Annotated, TypedDict
from dotenv import load_dotenv

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import Agents
from agents.researcher import run_researcher
from agents.critic import run_critic
from agents.writer import run_writer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(TypedDict):
    question: str
    pdf_path: str
    research_results: List[Dict[str, Any]]  # Snippets: [{"text": str, "page": int, "source": str}]
    research_queries: List[str]             # List of all queries run by researcher
    critic_feedback: str                    # Constructive feedback if rejected
    report: str                             # Synthesized report (markdown)
    loop_count: int                         # Active loop count to prevent infinite loop
    max_loops: int                          # Max corrections allowed
    history: List[Dict[str, Any]]           # Trace audit log of decisions
    llm_provider: str                       # 'openai' or 'gemini'
    llm_model: str                          # Specific LLM model name
    temperature: float                      # Generation temperature
    doc_type: str                           # e.g. "Exam Paper", "Financial Statement"
    doc_subject: str                        # e.g. "Object Oriented Programming - I"
    doc_title: str                          # Short descriptive title
    detailed_analysis_heading: str          # Custom heading for analysis section

def get_llm(provider: str, model_name: str | None = None, temperature: float = 0.0):
    """
    Helper function to load the specified LLM based on user UI choices.
    """
    provider_clean = provider.strip().lower()
    
    if provider_clean == "openai":
        from langchain_openai import ChatOpenAI
        model = model_name or "gpt-4o-mini"
        logger.info(f"Loading OpenAI LLM: {model} (temp: {temperature})")
        return ChatOpenAI(model=model, temperature=temperature, max_retries=6)
        
    elif provider_clean == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model_name or "gemini-2.5-flash"
        logger.info(f"Loading Google Gemini LLM: {model} (temp: {temperature})")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=6)
        
    else:
        # Auto-detect fallback
        if os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            logger.info("Auto-detected OpenAI. Loading gpt-4o-mini...")
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, max_retries=6)
        elif os.environ.get("GOOGLE_API_KEY"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            logger.info("Auto-detected Google Gemini. Loading gemini-2.5-flash...")
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, max_retries=6)
        else:
            raise ValueError(
                "No API keys configured. Set either OPENAI_API_KEY or GOOGLE_API_KEY in your .env "
                "or configure them in the Chainlit Chat Settings panel."
            )

from langchain_core.runnables import RunnableConfig

# Graph Node functions
async def researcher_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- ENTERING RESEARCHER NODE ---")
    # Dynamically fetch the model based on State setup
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("llm_model"),
        temperature=state.get("temperature", 0.0)
    )
    vector_store = config.get("configurable", {}).get("vector_store")
    result = await run_researcher(state, llm, vector_store)
    return result

async def critic_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- ENTERING CRITIC NODE ---")
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("llm_model"),
        temperature=state.get("temperature", 0.0)
    )
    result = await run_critic(state, llm)
    return result

async def writer_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- ENTERING WRITER NODE ---")
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("llm_model"),
        temperature=state.get("temperature", 0.2) # slightly higher temperature for synthesis
    )
    result = await run_writer(state, llm)
    return result

# Router function for self-correcting loop
def route_after_critic(state: AgentState) -> str:
    feedback = state.get("critic_feedback", "")
    
    if not feedback:
        logger.info("Critic approved. Directing workflow to Writer Agent.")
        return "writer"
    
    # If critic feedback exists, we need to loop back to the Researcher
    logger.info(f"Critic rejected. Routing back to Researcher Agent. Feedback details: {feedback}")
    return "researcher"

# Build and Compile the multi-agent graph
def build_agent_graph():
    logger.info("Building Agentic Graph workflow...")
    workflow = StateGraph(AgentState)  # type: ignore
    
    # Add Nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    
    # Set Entry Point
    workflow.set_entry_point("researcher")
    
    # Add Edges
    workflow.add_edge("researcher", "critic")
    
    # Add Conditional Edges
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "writer": "writer",
            "researcher": "researcher"
        }
    )
    
    workflow.add_edge("writer", END)
    
    # Compile graph with an in-memory memory saver checkpointer
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    logger.info("Workflow graph compiled successfully.")
    return compiled_graph

# Instantiate the graph
agent_graph = build_agent_graph()

async def run_multi_agent_system(
    question: str,
    pdf_path: str,
    vector_store: Any,
    session_id: str,
    settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes the multi-agent system for a given question and document.
    
    Args:
        question: User query.
        pdf_path: Path to the uploaded document.
        vector_store: Instantiated FAISS store.
        session_id: Thread ID for checkpointer tracking.
        settings: Configurations passed from the frontend UI.
        
    Returns:
        The final AgentState dictionary.
    """
    import chainlit as cl
    
    # Try to load classification metadata from user session
    doc_type = "General Document"
    doc_subject = "Unknown"
    doc_title = os.path.basename(pdf_path) if pdf_path else "Document"
    detailed_analysis_heading = "Detailed Analysis"
    
    user_session = None
    try:
        from chainlit.context import get_context
        get_context()
        user_session = cl.user_session
    except Exception:
        pass
        
    if user_session and user_session.get("doc_type"):
        doc_type = user_session.get("doc_type")
        doc_subject = user_session.get("doc_subject", "Unknown")
        doc_title = user_session.get("doc_title", os.path.basename(pdf_path))
        detailed_analysis_heading = user_session.get("detailed_analysis_heading", "Detailed Analysis")
    else:
        # Fallback to on-the-fly classification if pdf_path is provided
        if pdf_path and os.path.exists(pdf_path):
            try:
                from tools.pdf_parser import extract_pdf_sample
                from pydantic import BaseModel, Field
                
                import asyncio
                sample_text = await asyncio.to_thread(extract_pdf_sample, pdf_path, 800)
                
                class DocumentClassification(BaseModel):
                    document_type: str = Field(
                        description="The type of document, e.g., 'Financial Statement', 'Exam Paper', 'Academic Paper', 'Legal Contract', 'Technical Manual', 'General Document'."
                    )
                    organization: str = Field(
                        description="The organization that issued the document, e.g., 'Gujarat Technological University', 'Tesla Inc.', 'Unknown'."
                    )
                    subject: str = Field(
                        description="The main subject or topic of the document, e.g., 'Object Oriented Programming - I', 'Annual financial performance', 'Employment Agreement'."
                    )
                    document_title: str = Field(
                        description="A short, professional title summarizing this specific document."
                    )
                    detailed_analysis_heading: str = Field(
                        description="The most appropriate markdown heading for the detailed analysis section of this document type, e.g., 'Detailed Financial Analysis', 'Exam Paper Breakdown', 'Question Analysis', 'Contract Clause Summary', 'Technical Specifications Analysis'."
                    )
                
                llm = get_llm(
                    provider=settings.get("llm_provider", "gemini"),
                    model_name=settings.get("llm_model"),
                    temperature=0.0
                )
                
                classification_system_prompt = (
                    "You are an expert Document Classifier. Analyze the provided sample text of an uploaded document "
                    "and extract the document type, organization, subject, title, and the most appropriate section heading "
                    "for a detailed analysis of this document.\n\n"
                    "Response format: You MUST return a JSON matching the structured schema."
                )
                classification_user_prompt = f"Document Name: {os.path.basename(pdf_path)}\n\nSample Content:\n{sample_text}\n"
                
                structured_llm = llm.with_structured_output(DocumentClassification)
                doc_class = await structured_llm.ainvoke([
                    {"role": "system", "content": classification_system_prompt},
                    {"role": "user", "content": classification_user_prompt}
                ])
                
                if isinstance(doc_class, dict):
                    doc_type = str(doc_class.get("document_type", "General Document"))
                    doc_subject = str(doc_class.get("subject", "Unknown"))
                    doc_title = str(doc_class.get("document_title", os.path.basename(pdf_path)))
                    detailed_analysis_heading = str(doc_class.get("detailed_analysis_heading", "Detailed Analysis"))
                elif doc_class is not None:
                    doc_type = str(getattr(doc_class, "document_type", "General Document"))
                    doc_subject = str(getattr(doc_class, "subject", "Unknown"))
                    doc_title = str(getattr(doc_class, "document_title", os.path.basename(pdf_path)))
                    detailed_analysis_heading = str(getattr(doc_class, "detailed_analysis_heading", "Detailed Analysis"))
                else:
                    doc_type = "General Document"
                    doc_subject = "Unknown"
                    doc_title = os.path.basename(pdf_path)
                    detailed_analysis_heading = "Detailed Analysis"
                
                if user_session:
                    user_session.set("doc_type", doc_type)
                    user_session.set("doc_subject", doc_subject)
                    user_session.set("doc_title", doc_title)
                    user_session.set("detailed_analysis_heading", detailed_analysis_heading)
            except Exception as class_err:
                logger.warning(f"On-the-fly classification failed: {str(class_err)}")

    initial_state = {
        "question": question,
        "pdf_path": pdf_path,
        "research_results": [],
        "research_queries": [],
        "critic_feedback": "",
        "report": "",
        "loop_count": 0,
        "max_loops": int(settings.get("max_loops", 3)),
        "history": [],
        "llm_provider": settings.get("llm_provider", "openai"),
        "llm_model": settings.get("llm_model", ""),
        "temperature": float(settings.get("temperature", 0.0)),
        "doc_type": doc_type,
        "doc_subject": doc_subject,
        "doc_title": doc_title,
        "detailed_analysis_heading": detailed_analysis_heading
    }
    
    config = {
        "configurable": {
            "thread_id": session_id,
            "vector_store": vector_store
        }
    }
    
    logger.info(f"Invoking graph with session: {session_id}")
    final_state = await agent_graph.ainvoke(initial_state, config=config)
    return final_state
