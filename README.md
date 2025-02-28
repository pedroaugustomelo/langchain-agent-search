# GenAI Agents Challenge

This repository contains a reference implementation for the proposed challenge using **LangChain** and **LangGraph**. The challenge requires creating two GenAI agents and exposing them via a public API built with **Flask**. The solution must run within a Docker container orchestrated by Kubernetes, with an option for deployment on AWS. The two agents are:

1. **Conversational Agent**  
   - Responds to the user’s queries on any topic **except** Civil Engineering.  
   - Delegates to the Search Agent when it needs more information.  

2. **Search Agent**  
   - Performs up to 10 search operations when triggered by the Conversational Agent.  
   - Never responds directly to the user; it only provides findings back to the Conversational Agent.
  
## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Technologies Used](#technologies-used)
- [Deployment](#deployment)
  - [Current Deployment](#current-deployment)
  - [AWS Deployment Example](#aws-deployment-example)
- [License](#license)

  ## Features

- **Two-Agent System**:
  - **Conversational Agent** using large language models to chat with users on any topic besides Civil Engineering.
  - **Search Agent** to perform a maximum of 10 search results when more context or data is needed.

- **Flask API**:
  - A simple REST API that accepts user questions and returns responses from the Conversational Agent only.

- **Moderation**:
  - Uses **LlamaGuard** for content moderation.  
  - Provides added safety by filtering out disallowed topics, including the restricted domain of Civil Engineering.

- **Scalability**:
  - Designed to run in a **Docker** container within a **Kubernetes** cluster for high availability and easy scaling.

- **Extendibility**:
  - Integrations with **LangChain** and **LangGraph** can be extended to add more sophisticated prompt management, memory, or external APIs to enrich the conversation and search capabilities.

## Architecture Overview

                ┌─────────────────────────────┐
                │         User Client         │
                └─────────────┬───────────────┘
                              │
                              ▼
                 ┌─────────────────────────────┐
                 │         Flask API           │
                 │  (Pods in K8s Cluster)      │
                 └──────┬──────────────────────┘
                        │
               ┌────────▼────────┐
               │ Conversational  │
               │     Agent       │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   Search Agent  │
               │ (max 10 results)│
               └─────────────────┘

- **Conversational Agent**: Main interface for users’ questions. Decides whether to call the Search Agent.  
- **Search Agent**: Provides up to 10 search results from external or internal data sources. Returns these results to the Conversational Agent only when called.  
- **Flask API**: Exposes an endpoint (e.g., `/ask`) to receive user queries and routes them to the Conversational Agent.  
- **Kubernetes**: Orchestrates container deployments, ensuring scalability and reliability.  
- **LlamaGuard**: Ensures content moderation before generating final responses.

## Technologies Used

- **Python 3.x**
- **Flask** for building the REST API
- **LangChain** and **LangGraph** for building advanced AI workflows
- **Docker** for containerizing the application
- **Kubernetes** for container orchestration
- **AWS** (Optional) for deploying the containerized solution in a managed environment
- **LlamaGuard** (or similar) for content moderation
- **Other Tools**: Tools or APIs for external data searches

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software, subject to the terms detailed in the license.


