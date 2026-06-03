import os
import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class CriticEvaluation(BaseModel):
    approved: bool = Field(
        description="True if the retrieved context contains sufficient, relevant, and consistent information to fully answer the question. False otherwise."
    )
    feedback: str = Field(
        description="Brief feedback. If rejected, specify exactly what is missing in under 2 sentences. If approved, write 'Context approved'. Keep it very short to save API tokens."
    )

async def run_critic(state: Any, llm: Any) -> Dict[str, Any]:
    """
    Critic Agent Node.
    Reviews retrieved contexts to check for relevance, sufficiency, and consistency.
    If the context is incomplete, triggers a loop back with specific feedback.
    """
    import chainlit as cl
    
    question = state.get("question", "")
    research_results = state.get("research_results", [])
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    doc_type = state.get("doc_type", "General Document")
    doc_subject = state.get("doc_subject", "Unknown")
    
    step_name = "⚖️ Critic is verifying..."
    logger.info(f"Critic starting. Loop iteration: {loop_count + 1}/{max_loops} (Doc Type: {doc_type})")
    
    # Format retrieved passages for the prompt
    context_str = ""
    if not research_results:
        context_str = "[No context retrieved]"
    else:
        for idx, chunk in enumerate(research_results):
            context_str += f"--- Passage {idx+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
            context_str += f"{chunk['text']}\n\n"
            
    system_prompt = (
        f"You are an autonomous Quality Assurance and Critique Agent specializing in {doc_type} documents.\n"
        f"Your task is to judge whether the retrieved document context is SUFFICIENT and RELEVANT "
        f"to fully and accurately answer the user's question about the document (Subject/Topic: {doc_subject}). "
        f"You must prevent hallucinations and assumptions.\n\n"
        "Rules:\n"
        "1. Compare the user's question to the retrieved passages.\n"
    )
    
    if "financial" in doc_type.lower():
        system_prompt += (
            "2. For financial questions: check if specific years, numbers, calculations, or metrics are present "
            "in the context. If they are absent, you must REJECT (approved = false).\n"
        )
    elif "exam" in doc_type.lower() or "paper" in doc_type.lower() or "test" in doc_type.lower():
        system_prompt += (
            "2. For academic/exam questions: check if the exact questions, marks, parts, or instructions referenced "
            "in the user query are found in the context. If the details of the exam question or requirements are missing, "
            "you must REJECT (approved = false).\n"
        )
    else:
        system_prompt += (
            "2. Check if the specific facts, definitions, clauses, or instructions requested by the user's question "
            "are explicitly present in the context. If absent, you must REJECT (approved = false).\n"
        )
        
    system_prompt += (
        "3. Lenient for Simple Factual Questions: If the question is a direct lookup (e.g., 'how many', 'what is', 'when', 'who'), "
        "and the retrieved passages contain relevant information to answer it, you MUST approve (approved = true). "
        "Do NOT reject for missing peripheral details. Prevent unnecessary search loops to save time and API tokens.\n"
        "4. If the context is vague, incomplete, or does not address the question, REJECT.\n"
        "5. If you REJECT, write constructive, highly specific feedback detailing what is missing and "
        "what queries/keywords the Researcher should try next to find the data.\n"
        "6. If the context contains all necessary facts to answer the question comprehensively, APPROVE (approved = true).\n"
        "7. Do not hallucinate information. If details are missing, do not attempt to answer the question yourself.\n\n"
        "Response format: You MUST return a JSON object with 'approved' (boolean) and 'feedback' (string) keys."
    )
    
    user_prompt = (
        f"User Question: {question}\n\n"
        f"Retrieved Document Passages:\n{context_str}\n\n"
        f"Evaluate sufficiency."
    )
    
    evaluation = CriticEvaluation(approved=True, feedback="Context seems sufficient.")
    try:
        try:
            structured_llm = llm.with_structured_output(CriticEvaluation)
            evaluation = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            logger.warning(f"Critic structured output failed: {str(e)}.")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error("Rate limit hit during critic evaluation. Forcing approval to save API calls.")
                evaluation = CriticEvaluation(approved=True, feedback="Rate limit reached. Automatically approving context.")
                # Empty content to skip manual JSON parsing
                content = ""
            else:
                logger.info("Falling back to manual JSON parsing.")
                if "exam" in doc_type.lower() or "paper" in doc_type.lower():
                    example_feedback = "Missing Question 2(b) content. Try searching for 'Question 2' or specific subject terms."
                else:
                    example_feedback = "Missing specific document details. Try searching for relevant keywords from the question."
                    
                response = await llm.ainvoke([
                    SystemMessage(content=system_prompt + f"\nExample: {{\"approved\": false, \"feedback\": \"{example_feedback}\"}}"),
                    HumanMessage(content=user_prompt)
                ])
                content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            if content:
                data = json.loads(content)
                evaluation = CriticEvaluation(
                    approved=bool(data.get("approved", True)),
                    feedback=str(data.get("feedback", "No feedback provided."))
                )
    except Exception as e:
        logger.error(f"Critic evaluation error: {str(e)}. Defaulting to approval to avoid deadlocks.")
        evaluation = CriticEvaluation(approved=True, feedback=f"Critic error: {str(e)}. Forcing approval.")

    # Override approval if we hit max loops to prevent infinite loops
    final_approved = evaluation.approved
    critic_feedback = evaluation.feedback
    
    if not final_approved and loop_count >= (max_loops - 1):
        logger.warning(f"Max loop count ({max_loops}) reached. Forcing approval despite critic rejection.")
        final_approved = True
        critic_feedback = (
            f"[Max Correction Loops Reached] Original rejection feedback: {evaluation.feedback}. "
            "Proceeding with the best available information."
        )

    # Check if Chainlit is running active
    is_chainlit_active = False
    try:
        from chainlit.context import get_context
        get_context()
        is_chainlit_active = True
    except Exception:
        pass

    # UI Feedback Trace
    if is_chainlit_active:
        async with cl.Step(name=step_name, type="run") as step:  # type: ignore
            step.input = f"Question: {question}\nNumber of retrieved passages: {len(research_results)}\nLoop iteration: {loop_count + 1}/{max_loops}"
            status_symbol = "✅ Approved" if final_approved else "❌ Rejected / Self-Correcting"
            output_md = f"### Decision: {status_symbol}\n\n"
            output_md += f"**Critic Analysis & Feedback**:\n{critic_feedback}\n"
            step.output = output_md
        
    logger.info(f"Critic Finished. Approved: {final_approved}. Loop count: {loop_count}")

    # Return updated state
    return {
        "critic_feedback": "" if final_approved else critic_feedback,
        "loop_count": loop_count + (0 if final_approved else 1),
        "history": [{
            "loop": loop_count,
            "approved": final_approved,
            "feedback": critic_feedback,
            "retrieved_count": len(research_results)
        }]
    }
