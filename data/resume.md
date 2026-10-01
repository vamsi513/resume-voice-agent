<!--
Inspectable knowledge source for the voice agent.

Derived from data/resume_source.pdf (Vamsi_Sadu_Resume_AppliedAI_.pdf). It is the ONLY
factual source the agent may use. Nothing here was invented; `scripts/verify_source.py`
checks that every substantive term in the PDF is represented here and flags any term
here that has no source in the PDF.

Structure drives chunking (see app/chunking.py):
  ##   section  -> one chunk, when the section has no entries
  ###  entry    -> one chunk per project / role / degree
  "retrievable: false" in a section's preceding comment excludes it from the index.
-->

# Vamsi Krishna Sadu

<!-- retrievable: false — phone number and email address are deliberately NOT indexed.
     The agent should not read personal contact details aloud to an unknown caller; it
     answers "the resume doesn't provide that detail" instead.

     The real values are redacted here because this repository is public. They are NOT
     needed to run anything: this section is excluded from the index, so the agent never
     retrieves it either way. Location and relocation ARE indexed below, since those are
     ordinary professional facts a recruiter may ask about. -->
## Contact
[phone redacted] | [email redacted] | LinkedIn | GitHub

## Location
Based in Denton, TX. Open to relocation to San Jose, CA.

<!-- retrievable: false — the summary paraphrases every other section, so it matched
     everything and nothing: it took rank 1 on six questions whose real answer lived in
     a detail entry. Excluding it lifted Recall@1 from 0.778 to 0.889 and MRR from 0.868
     to 0.931 on eval/retrieval_set.json. Broad "tell me about him" questions are still
     answered, from the detail entries. Kept here because the PDF contains it. -->
## Summary
Applied AI engineer with an MS in Computer Science from the University of North Texas, who builds AI/ML applications, agents, and workflow automations end to end, from prototype through deployment and evaluation.
Built and deployed a LangGraph-based multi-step agentic research assistant to Kubernetes, load-tested under real traffic, and a production RAG system connecting LLMs to a knowledge base via sparse and dense retrieval.
Defined evaluation approaches and diagnosed failure modes across an LLM-judge platform benchmarking OpenAI, Anthropic, and a self-hosted Ollama judge, and built a real Model Context Protocol (MCP) tool-calling agent.
Full-stack across Python/FastAPI backends and Next.js/TypeScript frontends, with AWS cloud experience and a peer-reviewed publication (LREC 2026).

## Technical Skills
AI/LLM Systems: LangChain, LangGraph, LlamaIndex, AI Agents, MCP (Model Context Protocol), Retrieval-Augmented Generation (RAG), LLM Evaluation (RAGAS).
Vector Search & Retrieval: FAISS, Qdrant, Pinecone, BM25, Hybrid Retrieval (RRF), Cross-Encoder Reranking.
ML Frameworks: PyTorch, TensorFlow, Scikit-learn, HuggingFace Transformers, PEFT/LoRA.
Languages & Back-End: Python, Java, TypeScript, JavaScript, SQL, FastAPI, REST APIs, PostgreSQL, Redis.
MLOps & Cloud: MLflow, Docker, Kubernetes, CI/CD (GitHub Actions), AWS (EC2, S3, Bedrock), Google Cloud Platform (GCP).
Fundamentals & Practices: Object-Oriented Design, Testing, Debugging, Data Structures & Algorithms, Agile/Scrum, Git.

## Experience

### Graduate Research Assistant - LLM Evaluation, University of North Texas, Denton, TX (May 2025 - Aug 2025)
- Designed evaluation datasets, metrics, and model evaluation techniques to assess accuracy and reliability across GPT-4, Claude, Mistral, and LLaMA in 6 languages; co-authored the resulting paper, published in the Proceedings of LREC 2026.
- Built modular Python evaluation pipelines automating dataset preprocessing, prompt generation, and inference workflows, reducing manual evaluation time by an estimated 60%.

### Machine Learning Intern - Data Engineering & AI Pipelines, Codecasa, India (Aug 2023 - Jul 2024)
- Designed a Python/SQL ETL pipeline ingesting 500K+ daily enterprise records into a centralized feature store, reducing downstream model training latency by 30%.
- Created SQL views for downstream reporting and configured database access control; implemented automated data validation catching quality issues before they propagated downstream.

### Software Engineering Intern, Coding Raja Technologies, India (Jul 2022 - Jul 2023)
- Built e-commerce features end to end with a Django backend and a React frontend, integrating REST APIs across product, order, and payment data; debugged and resolved integration issues across the API layer.
- Collaborated in an agile team environment with sprint planning, daily standups, and retrospectives, using Git for version control.

## Projects

### AgentIQ - Multi-Step Agentic Research Assistant (Apr 2026 - Present)
Technologies: Python, LangGraph, FAISS, Pinecone, LlamaIndex, FastAPI, Next.js/TypeScript, Kubernetes (Red Hat OpenShift), AWS EC2.
- Built and owned a full-stack application end to end: a FastAPI back-end service with a Next.js (React-based) front-end, containerized with Docker and deployed to Kubernetes.
- Architected an agentic research assistant using LangGraph to orchestrate multi-step retrieval and reasoning, load-tested under real traffic (1 to 4 replicas across about 1,371 requests, zero crashes), with an automated RAGAS evaluation loop reaching 0.91 faithfulness and 0.84 answer relevancy.
- Source code is on GitHub.

### IncidentMemory AI - Production RAG System for Document Retrieval (Mar 2026 - Present)
Technologies: Python, FastAPI, Qdrant, BM25, sentence-transformers, MLflow, PostgreSQL, Redis, Docker, AWS EC2.
- Designed and deployed a Retrieval-Augmented Generation pipeline integrating sparse and dense retrievers (BM25 plus Qdrant vector search, fused via RRF, with cross-encoder reranking) over a vectorized knowledge base to generate contextual, grounded responses.
- Applied debugging and testing practices to catch and fix retrieval quality issues, tracked in MLflow (Recall@1, Recall@3, Recall@5, and MRR).
- Source code is on GitHub.

### MCP Tool-Calling Agent - LLM-Based Tool-Calling System (Sep 2026 - Present)
Technologies: Python, FastMCP (Model Context Protocol), stdio JSON-RPC, pytest, GitHub Actions CI.
- Built an LLM-based agent workflow using the Model Context Protocol: a server exposing 3 tools (GitHub repo search, URL fetch, and SQLite query) over stdio JSON-RPC, covered by a pytest suite running in CI for reliability.

### TriageTune - LoRA Fine-Tuning Experiment (Sep 2026 - Present)
Technologies: Python, NumPy, Pandas, Scikit-learn, PEFT/LoRA, HuggingFace Transformers.
- Built and evaluated four modeling approaches on the same task using bootstrap confidence intervals (10,000 samples); found a classical baseline still outperformed a LoRA-fine-tuned LLM by 4.2 points, an unexpected result reported honestly rather than a single number.

## Education

### Master of Science, Computer Science - University of North Texas, Denton, TX (Aug 2024 - May 2026)
- Degree date May 2026. GPA 3.909.
- Teaching Assistant, Algorithms - University of North Texas, Fall 2025 to Spring 2026.

## Publication
Cross-Lingual Stability and Bias in Instruction-Tuned Language Models for Humanitarian NLP.
Published in the Proceedings of LREC 2026, focused on fairness and bias mitigation in multilingual LLM behavior.

## Certifications
- Building with the Claude API - Anthropic, Mar 2026.
- Introduction to Model Context Protocol - Anthropic, Mar 2026.
