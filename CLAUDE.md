# RAG Knowledge Agent

## Project Overview

This is a portfolio project called **RAG Knowledge Agent**.

The goal is to build an AI assistant that answers user questions using a controlled internal knowledge base.

The assistant must retrieve relevant information from documents before answering and should avoid inventing information that is not supported by the knowledge base.

This project is being developed incrementally as part of a professional portfolio focused on AI Agents, Generative AI, RAG, Data, and Python.

## Main Goals

The application should eventually:

1. Ingest a small set of internal company documents.
2. Split documents into useful chunks.
3. Generate embeddings for those chunks.
4. Store and search document embeddings.
5. Retrieve relevant context for a user's question.
6. Generate answers grounded in retrieved context.
7. Provide source references or citations.
8. Refuse or qualify answers when the documents do not contain enough information.
9. Support questions in Spanish and English while using Spanish as the default response language.
10. Provide a simple user interface.
11. Include an evaluation framework for retrieval and answer quality.

## Language Policy

Spanish is the primary user-facing language of this project.

### User-facing content

The following must be written in Spanish:

- Application interface
- Buttons and labels
- Error messages
- Loading messages
- Example questions
- Knowledge-base documents
- Agent answers
- User-facing source citations
- Evaluation questions where practical

The agent should respond in Spanish by default.

If the user writes in English, the agent may understand the question but should still respond in Spanish unless the user explicitly asks for an English response.

### Technical content

The following should remain in English:

- Python code
- Function names
- Variable names
- Class names
- Module names
- Code comments
- Docstrings
- Git commit messages
- GitHub README

The README should remain in English because this repository is intended as a technical portfolio for international roles.

### Knowledge Base

The fictional SaaS knowledge-base documents should be written in natural Spanish.

Document titles, sections, policies, FAQs, procedures, pricing descriptions, and support documentation should all use Spanish.

The RAG system must preserve Spanish accents and characters correctly using UTF-8.

### Retrieval

Retrieval must work correctly with Spanish questions and Spanish source documents.

The embedding and retrieval architecture should not assume English-only text.

### Agent Behavior

The final RAG agent should:

1. Answer in Spanish by default.
2. Base its answers only on retrieved documentation.
3. Cite sources using their Spanish document titles.
4. Clearly state in Spanish when the available documentation is insufficient.
5. Never translate or invent policies merely to answer a question.

## Fictional Business Domain

The knowledge base will represent a fictional SaaS company.

Documents may include topics such as:

- Product overview
- Pricing and subscription plans
- Billing
- Cancellation policies
- Refund policies
- Account management
- Password recovery
- User onboarding
- Data privacy
- Security
- Support procedures
- Common troubleshooting
- Frequently asked questions

All fictional knowledge-base documents should be written in Spanish.

The documents must contain fictional information only.

Do not use confidential, copyrighted, or real company internal documentation.

## Planned Tech Stack

- Python 3.13+
- OpenAI API
- Embeddings
- Retrieval-Augmented Generation (RAG)
- Vector search
- Streamlit
- pytest
- Git / GitHub

Additional technologies should only be introduced when they solve a real project requirement.

Do not add LangChain, LangGraph, CrewAI, n8n, or other frameworks unless explicitly requested later.

Prefer simple, understandable implementations.

## Development Philosophy

This project should remain:

- Simple
- Modular
- Easy to understand
- Easy to explain during a technical interview
- Well documented
- Testable
- Grounded in source documents
- Suitable for a junior AI / Generative AI portfolio

Avoid overengineering.

Prefer straightforward Python implementations over unnecessary abstractions.

## Development Workflow

The project will be developed incrementally.

Do not implement future features unless explicitly requested.

Each iteration should focus on one clear piece of functionality.

Before making significant architectural changes:

1. Explain the proposed change.
2. Explain why it is necessary.
3. Prefer the simplest solution that satisfies the requirement.

Do not refactor unrelated code unless necessary.

## Code Quality

Follow these principles:

- Use clear and descriptive names.
- Keep functions small and focused.
- Use type hints where useful.
- Follow PEP 8 conventions.
- Avoid duplicated logic.
- Prefer readable code over clever code.
- Add comments only when they explain non-obvious decisions.

## RAG Principles

The assistant must follow these principles:

1. Retrieval should happen before generating a knowledge-based answer.
2. Answers should be grounded in retrieved document context.
3. The model should not invent policies, prices, features, or procedures.
4. If the retrieved context does not support an answer, the assistant should say in Spanish that the available documentation is insufficient.
5. Sources used for the answer should be visible to the user.
6. Retrieval and answer generation should remain separate, testable steps.
7. Citations must refer only to sources that were actually retrieved and used.
8. The system should prefer grounded incompleteness over a plausible but unsupported answer.

## Documents

Source documents should live in:

`knowledge/`

Documents should be simple and easy to inspect.

Prefer Markdown or plain-text files initially.

Each document should have:

- A clear Spanish title
- A clear topic
- Well-structured sections
- Fictional business information only

Documents must use UTF-8 encoding.

Do not commit confidential information or API credentials.

## Chunking

Document chunking should:

- preserve useful context
- avoid extremely large chunks
- avoid excessively small fragments
- be deterministic
- preserve source metadata
- preserve Spanish characters correctly

Each chunk should retain enough metadata to identify:

- source document
- section when available
- chunk identifier

Chunking should be independently testable.

## Embeddings

Embedding generation should be isolated behind a simple function or service layer.

Do not spread direct OpenAI embedding calls throughout the codebase.

The system should make it possible to test retrieval logic without making real API calls.

The selected embedding model must support multilingual semantic retrieval, including Spanish.

## Vector Search

Keep vector search simple for the MVP.

Prefer an implementation that is:

- understandable
- deterministic where practical
- easy to run locally
- suitable for a portfolio demo

Do not introduce a production vector database unless it becomes necessary.

The initial implementation should favor transparency over infrastructure complexity.

## Retrieval

Retrieval should return structured results.

Each retrieved result should contain at minimum:

- chunk text
- source document
- chunk identifier
- similarity score or equivalent retrieval score

Retrieval logic must remain independently testable from answer generation.

The system should make it possible to inspect which chunks were retrieved for a question.

## Grounding and Citations

The final answer must be based on retrieved chunks.

Answers should cite the documents that actually supported the response.

Where practical, citations should include:

- source document name
- section name or chunk metadata

Example style:

`Fuente: Política de Reembolsos — Reembolsos anuales`

Do not fabricate citations.

Do not cite a document unless it actually contributed retrieved context to the answer.

If the retrieved evidence is insufficient, the assistant should respond with a clear Spanish limitation message rather than attempting to infer an unsupported answer.

## Security and Secrets

Never:

- commit API keys
- hardcode secrets
- commit `.env`
- expose secrets in logs
- include real confidential documents
- expose full prompts if they contain secrets

Secrets must be stored in environment variables.

The repository should only contain:

`.env.example`

If Streamlit local secrets are introduced later, real secrets files must remain gitignored.

## Environment Variables

Expected variables may include:

- OPENAI_API_KEY
- OPENAI_MODEL
- OPENAI_EMBEDDING_MODEL

Additional configuration may be added later only when required.

Do not add environment variables speculatively.

## Testing

pytest will be used for testing.

Unit tests must not make real OpenAI API calls.

Use mocks, monkeypatching, or deterministic test doubles where appropriate.

Priority test areas include:

- document loading
- UTF-8 handling
- chunking
- metadata preservation
- embedding interfaces
- similarity search
- retrieval ranking
- retrieval metadata
- grounding behavior
- citation/source handling
- unsupported-question behavior
- multilingual input
- Spanish default responses
- UI helper logic
- evaluation framework

Real end-to-end evaluations may call the OpenAI API, but they must remain separate from pytest.

## Evaluation

The project should eventually include an evaluation suite separate from unit tests.

Useful evaluation dimensions include:

- Did retrieval return the correct document?
- Did retrieval return the correct chunk or section?
- Did the answer use retrieved evidence?
- Did the answer contain the expected factual information?
- Did the assistant avoid fabricating answers?
- Did the answer cite the correct source?
- Did unsupported questions receive an appropriate limitation response?
- Does an English question still receive a Spanish answer by default?
- Did the system avoid citing irrelevant documents?

Do not use an LLM-as-a-judge initially unless explicitly requested.

Prefer transparent, rule-based evaluation where possible.

## Observability

The project should eventually make relevant RAG activity inspectable.

Useful observable information may include:

- retrieved source documents
- retrieved chunk identifiers
- similarity scores
- number of retrieved chunks
- retrieval duration

Do not expose private chain-of-thought or hidden model reasoning.

Observability should show system/tool activity, not internal reasoning.

## Git Workflow

Changes should be small and logically grouped.

Use descriptive commits such as:

- `docs: add Claude project instructions`
- `docs: define project language policy`
- `chore: initialize project structure`
- `docs: add fictional knowledge base`
- `feat: add document loader`
- `feat: add deterministic text chunking`
- `feat: add embedding service`
- `feat: add vector retrieval`
- `feat: add grounded RAG agent`
- `feat: add retrieval observability`
- `test: add RAG evaluation framework`
- `feat: add Streamlit interface`
- `docs: polish portfolio README`

Do not automatically commit or push changes unless explicitly requested.

## Documentation

The final README should eventually explain:

- Problem
- Architecture
- RAG pipeline
- Knowledge base
- Chunking strategy
- Embeddings
- Retrieval
- Grounding
- Citations
- Observability
- Evaluation
- Tech stack
- Installation
- Environment setup
- Example questions
- Screenshots
- Security considerations
- Limitations
- Future improvements
- Live demo

The README must remain in English.

## Portfolio Quality

This repository is intended to be reviewed by recruiters and technical interviewers.

The final project should make it easy to understand:

1. What problem the application solves.
2. How the RAG pipeline works.
3. How hallucinations are reduced.
4. How retrieved evidence is used.
5. How citations are generated.
6. How retrieval quality is evaluated.
7. Which design decisions were made and why.

Do not exaggerate capabilities.

Avoid unsupported claims such as:

- production-ready
- enterprise-grade
- zero hallucinations
- perfect retrieval

Prefer factual descriptions backed by implemented behavior and tests.

## Important Instruction

Always respect the scope of the current task.

If asked to implement one feature, do not automatically implement later stages of the project.

At the end of each task, summarize:

1. What was changed.
2. Why it was changed.
3. Files created or modified.
4. How to test the changes.
5. Any relevant decisions or limitations.

Do not commit or push unless explicitly requested.