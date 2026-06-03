import os
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from tools.vector_store import query_vector_store

logger = logging.getLogger(__name__)

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 1-2 specific search queries to retrieve relevant document information.")

async def run_researcher(state: Any, llm: Any, vector_store: Any = None) -> Dict[str, Any]:
    """
    Researcher Agent Node.
    Generates search queries based on the user's question and any critic feedback,
    queries the vector store, and returns the retrieved document snippets.
    """
    import chainlit as cl
    
    question = state.get("question", "")
    critic_feedback = state.get("critic_feedback", "")
    
    step_name = "🕵️ Researcher is scanning..."
    logger.info(f"Researcher starting. Question: '{question}', Feedback: '{critic_feedback}'")
    
    # Build System Prompt
    system_prompt = (
        "You are an expert Document Researcher Agent. Your task is to generate 1 to 2 "
        "search queries to retrieve the exact information needed to answer the user's question.\n"
        "Analyze the user's question and any critic feedback. Focus your queries on retrieving "
        "specific numbers, facts, tables, statements, or notes that address the details requested.\n\n"
        "Response format: You MUST return a JSON object with a single key 'queries' containing a list of strings."
    )
    
    user_prompt = f"User Question: {question}\n"
    if critic_feedback:
        user_prompt += f"Critic Feedback (address this directly): {critic_feedback}\n"
    user_prompt += "\nGenerate the search queries."
    
    queries = []
    if not critic_feedback:
        logger.info("First research pass: using user's question directly as search query to save API tokens.")
        queries = [question]
    else:
        try:
            # Try structured output first
            try:
                structured_llm = llm.with_structured_output(SearchQueries)
                structured_response = await structured_llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                queries = structured_response.queries
            except Exception as e:
                logger.warning(f"Structured output failed: {str(e)}.")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.error("Rate limit hit during query generation. Falling back to raw inputs.")
                    queries = [question, critic_feedback[:100]]
                else:
                    logger.info("Falling back to manual JSON parsing.")
                    response = await llm.ainvoke([
                        SystemMessage(content=system_prompt + "\nExample: {\"queries\": [\"Q1\", \"Q2\"]}"),
                        HumanMessage(content=user_prompt)
                    ])
                    content = response.content.strip()
                    
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    data = json.loads(content)
                    queries = data.get("queries", [])
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            queries = [question]
            if critic_feedback:
                queries.append(critic_feedback[:100])

    # Check if Chainlit is running active
    is_chainlit_active = False
    try:
        from chainlit.context import get_context
        get_context()
        is_chainlit_active = True
    except Exception:
        pass

    # Perform searches and handle step rendering
    retrieved_chunks = []
    seen_texts = set()
    
    try:
        if is_chainlit_active:
            async with cl.Step(name=step_name, type="run") as step:  # type: ignore
                input_desc = f"**User Question**: {question}"
                if critic_feedback:
                    input_desc += f"\n\n**Critic Feedback**: {critic_feedback}"
                step.input = input_desc
                
                query_list_md = "\n".join([f"- `{q}`" for q in queries])
                await step.stream_token(f"### Generated Search Queries:\n{query_list_md}\n\n")
                
                if vector_store:
                    for query in queries:
                        results = query_vector_store(vector_store, query, k=4)
                        for doc in results:
                            content = doc.page_content.strip()
                            if content not in seen_texts:
                                seen_texts.add(content)
                                retrieved_chunks.append({
                                    "text": doc.page_content,
                                    "page": doc.metadata.get("page", "Unknown"),
                                    "source": doc.metadata.get("source", "Document")
                                })
                else:
                    await step.stream_token("⚠️ **Warning**: No document uploaded. Scanning skipped.\n")
                    
                output_md = f"Found {len(retrieved_chunks)} relevant passage(s) across the document pages.\n\n"
                if retrieved_chunks:
                    output_md += "### Sample of Retrieved Context:\n"
                    for idx, chunk in enumerate(retrieved_chunks[:2]):
                        snippet = chunk["text"][:180].replace("\n", " ") + "..."
                        output_md += f"- **[Page {chunk['page']}]**: *\"{snippet}\"*\n"
                step.output = output_md
        else:
            if vector_store:
                for query in queries:
                    results = query_vector_store(vector_store, query, k=4)
                    for doc in results:
                        content = doc.page_content.strip()
                        if content not in seen_texts:
                            seen_texts.add(content)
                            retrieved_chunks.append({
                                "text": doc.page_content,
                                "page": doc.metadata.get("page", "Unknown"),
                                "source": doc.metadata.get("source", "Document")
                            })
    except Exception as e:
        logger.error(f"Error during document search: {str(e)}")

    # Log queries
    research_queries = state.get("research_queries", [])
    research_queries.extend(queries)
    
    return {
        "research_results": retrieved_chunks,
        "research_queries": research_queries,
        "critic_feedback": ""
    }
