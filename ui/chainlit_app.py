import os
import sys
import logging
import shutil
import chainlit as cl

from dotenv import load_dotenv
load_dotenv(override=True)

# Resolve parent directory to allow correct imports when running the script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import run_multi_agent_system

logger = logging.getLogger(__name__)

async def build_index(pdf_path: str, status_msg: cl.Message):
    """
    Chunks the uploaded PDF and indexes it into FAISS.
    """
    user_session = cl.user_session
    settings = (user_session.get("settings") if user_session else None) or {}
    emb_provider = str(settings.get("embedding_provider", "local"))
    
    status_msg.content = "⚙️ Extracting and chunking document text..."
    await status_msg.update()
    
    try:
        import asyncio
        from tools.pdf_parser import extract_documents_from_pdf
        
        # Run text extraction asynchronously in thread pool to prevent blocking Chainlit event loop
        chunks = await asyncio.to_thread(extract_documents_from_pdf, pdf_path)
        
        if not chunks:
            status_msg.content = "❌ **Error**: No readable text was extracted from this PDF. Please check if the document is scanned or password protected."
            await status_msg.update()
            return
            
        status_msg.content = f"⚙️ Initializing `{emb_provider}` embedding model..."
        await status_msg.update()
        
        from tools.vector_store import get_embeddings, create_vector_store
        
        # Provide fallback values to avoid None type check complaints
        openai_key = os.environ.get("OPENAI_API_KEY") or ""
        google_key = os.environ.get("GOOGLE_API_KEY") or ""
        
        embeddings = get_embeddings(
            provider=emb_provider,
            openai_api_key=openai_key,
            google_api_key=google_key
        )
        
        status_msg.content = f"⚙️ Building in-memory FAISS index for {len(chunks)} text chunks..."
        await status_msg.update()
        
        # Run FAISS index creation asynchronously in a thread pool
        vector_store = await asyncio.to_thread(create_vector_store, chunks, embeddings)
        if user_session:
            user_session.set("vector_store", vector_store)
        
        # Dynamic document type classification
        status_msg.content = "⚙️ Classifying document type..."
        await status_msg.update()
        
        try:
            from tools.pdf_parser import extract_pdf_sample
            from app import get_llm
            from pydantic import BaseModel, Field
            
            # Extract sample text asynchronously
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

            # Initialize LLM for classification using settings
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
            
            # Call using structured output
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
            
        except Exception as class_err:
            logger.warning(f"Failed to classify document: {str(class_err)}. Using default classification.")
            doc_type = "General Document"
            doc_subject = "Unknown"
            doc_title = os.path.basename(pdf_path)
            detailed_analysis_heading = "Detailed Analysis"
            
        if user_session:
            user_session.set("doc_type", doc_type)
            user_session.set("doc_subject", doc_subject)
            user_session.set("doc_title", doc_title)
            user_session.set("detailed_analysis_heading", detailed_analysis_heading)
        
        success_info = f"✅ **{os.path.basename(pdf_path)}** successfully processed and indexed ({len(chunks)} chunks).\n\n"
        success_info += f"📁 **Detected**: {doc_type}\n"
        if doc_subject and doc_subject != "Unknown":
            success_info += f"📚 **Subject/Topic**: {doc_subject}\n"
        success_info += f"\nAsk me anything about this document!"
        
        status_msg.content = success_info
        await status_msg.update()
    except Exception as e:
        logger.error(f"Error building vector index: {str(e)}", exc_info=True)
        status_msg.content = (
            f"❌ **Error indexing document**: {str(e)}\n\n"
            f"**Troubleshooting**:\n"
            f"- Verify API keys in your environment if using OpenAI/Gemini embeddings.\n"
            f"- Try changing the 'Embedding Model Provider' to `local` in the Chat Settings panel (bottom-left) to run offline."
        )
        await status_msg.update()

async def upload_pdf_dialog():
    """
    Displays the PDF upload prompt and processes the uploaded file.
    """
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a financial document (PDF) to start.",
            accept=["application/pdf"],
            max_size_mb=30,
            timeout=300
        ).send()
        
    file = files[0]
    
    # Save document locally
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, file.name)
    
    # Copy from Chainlit temporary path
    shutil.copy(file.path, pdf_path)
        
    user_session = cl.user_session
    if user_session:
        user_session.set("pdf_path", pdf_path)
        user_session.set("pdf_name", file.name)
    
    status_msg = cl.Message(content=f"⚙️ Saving `{file.name}`...")
    await status_msg.send()
    
    await build_index(pdf_path, status_msg)

@cl.on_chat_start
async def start():
    """
    Chainlit session startup. Set widgets, show welcome screen, and request file.
    """
    # Configure user settings widget
    settings = await cl.ChatSettings([
        cl.input_widget.Select(
            id="llm_provider",
            label="LLM Provider",
            values=["gemini", "openai"],
            initial_index=0
        ),
        cl.input_widget.TextInput(
            id="llm_model",
            label="LLM Model (optional / overrides default)",
            initial=""
        ),
        cl.input_widget.Slider(
            id="temperature",
            label="LLM Temperature (0.0=Deterministic, 1.0=Creative)",
            initial=0.0,
            min=0.0,
            max=1.0,
            step=0.1
        ),
        cl.input_widget.Slider(
            id="max_loops",
            label="Max Self-Correction Loops",
            initial=3,
            min=1,
            max=5,
            step=1
        ),
        cl.input_widget.Select(
            id="embedding_provider",
            label="Embedding Model Provider",
            values=["local", "openai", "gemini"],
            initial_index=0
        )
    ]).send()
    
    user_session = cl.user_session
    if user_session:
        user_session.set("settings", settings)
    
    # Welcome banner using beautiful Markdown structure
    welcome_markdown = (
        "# 🕵️‍♂️ VeriDoc AI\n"
        "### *Self-Correcting Multi-Agent Financial Document Analyst*\n\n"
        "Experience a production-grade multi-agent agentic workflow. "
        "A **Researcher** extracts context, a **Critic** audits for hallucinations "
        "and completeness, and a **Writer** drafts a structured, cited executive report."
    )
    welcome_msg = cl.Message(content=welcome_markdown)
    setattr(welcome_msg, "disable_feedback", True)
    await welcome_msg.send()
    
    # Prompt the user to upload a document
    await upload_pdf_dialog()

@cl.on_settings_update
async def on_settings_update(settings):
    """
    Rebuilds vector index if the embedding provider changes during active chat.
    """
    user_session = cl.user_session
    old_settings = (user_session.get("settings") if user_session else None) or {}
    if user_session:
        user_session.set("settings", settings)
    
    pdf_path = user_session.get("pdf_path") if user_session else None
    
    # Check if the embedding provider has changed
    if old_settings.get("embedding_provider") != settings.get("embedding_provider"):
        if pdf_path:
            status_msg = cl.Message(content="🔄 Embedding provider changed. Re-indexing document...")
            await status_msg.send()
            await build_index(pdf_path, status_msg)
        else:
            await cl.Message(content="⚙️ Settings updated. Upload a document to index it.").send()
    else:
        await cl.Message(content="⚙️ Settings updated.").send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handles user questions, executes the LangGraph workflow, and posts findings.
    """
    user_session = cl.user_session
    pdf_path = user_session.get("pdf_path") if user_session else None
    vector_store = user_session.get("vector_store") if user_session else None
    settings = (user_session.get("settings") if user_session else None) or {}
    
    # Validate that document is uploaded
    if not pdf_path or not vector_store:
        msg = cl.Message(content="⚠️ **No document found**. Please upload a PDF before asking questions.")
        await msg.send()
        await upload_pdf_dialog()
        return

    # Check if API keys are set for chosen provider
    provider = str(settings.get("llm_provider", "openai")).lower()
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        await cl.Message(content="⚠️ **OpenAI API Key is missing**. Please set `OPENAI_API_KEY` in your environment or select Gemini provider.").send()
        return
    if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        await cl.Message(content="⚠️ **Google API Key is missing**. Please set `GOOGLE_API_KEY` in your environment or select OpenAI provider.").send()
        return

    session_id = str(user_session.get("id") if user_session else "default_session")
    
    try:
        # Run LangGraph Orchestrator
        final_state = await run_multi_agent_system(
            question=message.content,
            pdf_path=pdf_path,
            vector_store=vector_store,
            session_id=session_id,
            settings=settings
        )
        
        report = final_state.get("report", "")
        research_results = final_state.get("research_results", [])
        
        # Prepare side-panel UI elements for citations
        text_elements = []
        if research_results:
            for idx, chunk in enumerate(research_results):
                element_name = f"Source {idx+1} [Page {chunk['page']}]"
                text_elements.append(
                    cl.Text(
                        name=element_name,
                        content=chunk["text"],
                        display="side"
                    )
                )
                
        if not report:
            report = "❌ The writer agent was unable to synthesize the report. Please try refining your question."

        # Send final report to user along with side-panel text citations
        await cl.Message(
            content=report,
            elements=text_elements
        ).send()

    except Exception as e:
        logger.error(f"Error during graph execution: {str(e)}", exc_info=True)
        await cl.Message(content=f"❌ **System Error during processing**: {str(e)}").send()
