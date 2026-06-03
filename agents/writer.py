import os
import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

async def run_writer(state: Any, llm: Any) -> Dict[str, Any]:
    """
    Writer Agent Node.
    Synthesizes the validated context into a comprehensive, highly-formatted financial report.
    Ensures precise inline citations referencing the page numbers of the source text.
    """
    import chainlit as cl
    
    question = state.get("question", "")
    research_results = state.get("research_results", [])
    history = state.get("history", [])
    doc_type = state.get("doc_type", "General Document")
    doc_subject = state.get("doc_subject", "Unknown")
    doc_title = state.get("doc_title", "Document")
    detailed_analysis_heading = state.get("detailed_analysis_heading", "Detailed Analysis")
    
    step_name = "✍️ Writer is formatting..."
    logger.info(f"Writer starting. Generating report for query: '{question}' (Doc Type: {doc_type})")
    
    # Format the verified context
    context_str = ""
    for idx, chunk in enumerate(research_results):
        context_str += f"--- Source Chunk {idx+1} (Document: {chunk['source']}, Page: {chunk['page']}) ---\n"
        context_str += f"{chunk['text']}\n\n"

    # Determine if the query is requesting a full detailed report
    is_detailed_request = any(word in question.lower() for word in ["report", "analysis", "summar", "detailed", "breakdown", "compile"])
    
    if is_detailed_request:
        system_prompt = (
            f"You are an Elite Document Analyst and Professional Writer specializing in {doc_type} documents.\n"
            f"Your goal is to synthesize the provided document context into a professional, executive-ready "
            f"analytical report that directly answers the user's question about the document '{doc_title}'.\n\n"
            "Writing Guidelines:\n"
            "1. Structure: Format the report beautifully using clean markdown. Use clear sections, bullet points, "
            "and tables/lists where applicable.\n"
            "2. Structure Outline:\n"
            "   - **Title**: A professional report title reflecting the document context\n"
            "   - **Executive Summary**: A high-level overview of the answer\n"
            f"   - **{detailed_analysis_heading}**: Deep-dive analysis. You MUST use exactly this heading for the detailed analysis section.\n"
            "   - **Verified Sources & Citations**: List of documents, page numbers, and brief referenced quotes\n"
            "3. Precise Citations: You MUST insert inline citations (e.g. `[Page X]` or `[DocName, Page X]`) "
            "immediately adjacent to any numbers, data points, or key claims. Specify the page and the specific part "
            "of the page (e.g., table, paragraph, header).\n"
            "4. Highlight the Final Answer: Clearly highlight the final, key answer using Markdown bold (e.g. **Answer: 1 student**).\n"
            "5. Strict Grounding: Rely ONLY on the provided context. If information is requested but missing, "
            "explicitly state that the data was not found in the uploaded document. Do not invent any numbers or facts.\n"
            "6. Tone: Objective, analytical, formal, and precise."
        )
    else:
        system_prompt = (
            f"You are an Elite Document Analyst and Professional Writer specializing in {doc_type} documents.\n"
            f"Your goal is to answer the user's question about the document '{doc_title}' as CONCISELY and DIRECTLY as possible.\n\n"
            "Writing & Formatting Guidelines:\n"
            "1. Concise Answer: Give only the direct answer. Do NOT include boilerplate sections like 'Executive Summary', "
            "'Detailed Analysis', or any headers unless specifically requested. Do not output extra information that was not asked.\n"
            "2. Specific Source Citation: For every fact, specify exactly which page and which section/part of the page "
            "(e.g., 'Page 1, table in the middle', 'Page 3, paragraph 2', 'Page 1, header section') the answer was taken from.\n"
            "3. Highlight the Final Answer: You MUST highlight the final, key answer clearly. Use Markdown bold "
            "(e.g. **Answer: 1 student**) or a prominent blockquote.\n"
            "4. Strict Grounding: Rely ONLY on the provided context. Do not invent or assume anything."
        )
    
    # Check if we bypassed critic rejections due to loops limit
    loop_notes = ""
    if history and not history[-1].get("approved", True):
        loop_notes = "\nNote: The Critic rejected the latest research due to missing details, but we reached the loop limit. Address the question to the best of your ability with the available text and clearly note any gaps or limitations."

    if is_detailed_request:
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Verified Context Chunks:\n{context_str}\n"
            f"{loop_notes}\n\n"
            f"Draft the professional analysis report now."
        )
    else:
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Verified Context Chunks:\n{context_str}\n"
            f"{loop_notes}\n\n"
            f"Draft the concise direct answer now."
        )
    
    # Check if Chainlit is running active
    is_chainlit_active = False
    try:
        from chainlit.context import get_context
        get_context()
        is_chainlit_active = True
    except Exception:
        pass
 
    report_content = ""
    try:
        if is_chainlit_active:
            async with cl.Step(name=step_name, type="run") as step:  # type: ignore
                step.input = f"Drafting report for question: '{question}' using {len(research_results)} verified contexts."
                await step.stream_token("### Drafting Report...\n\n")
                async for chunk in llm.astream([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]):
                    await step.stream_token(chunk.content)
                    report_content += chunk.content
        else:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            report_content = response.content
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        report_content = f"### Document Analysis Report\n\nFailed to generate report due to an LLM error: {str(e)}"
            
    return {
        "report": report_content
    }
