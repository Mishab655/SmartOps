# SmartOps
Live link: https://smart-ops-rosy.vercel.app/
**SmartOps** is an AI-powered e-commerce chatbot and operational assistant designed to provide intelligent insights, data retrieval, and strategic decision-making capabilities. 

It leverages an advanced multi-agent architecture to process user queries, interact with both structured (SQL) and unstructured (Vector) databases, and deliver actionable business intelligence.

## Key Features

- **Dynamic Query Routing**: Intelligently routes queries based on user intent (forecast, sentiment, churn, strategy, retrieval, general).
- **Multi-Agent Architecture**: Uses specialized agents for distinct tasks:
  - **Forecast Agent**: Predicts sales and demand for specific product categories.
  - **Sentiment Agent**: Analyzes customer review scores and feedback.
  - **Churn Agent**: Evaluates customer retention metrics and identifies churn risks.
  - **Decision Agent**: Generates rule-based strategic actions by combining data from Forecast and Sentiment agents.
  - **Retrieval Agent**: Handles complex text-to-SQL tasks to query the primary relational database.
  - **RAG Agent**: Uses Retrieval-Augmented Generation to answer general or meta-questions.
- **High-Performance UI**: Custom HTML/CSS/JavaScript frontend with robust chat history and persistent note-taking.
- **Lazy-Loaded Modules**: Optimized memory usage and startup times by loading AI agents only when needed.

## Technology Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: Custom HTML / CSS / Vanilla JavaScript (deployed on Vercel)
- **AI & LLMs**: 
  - Provider: Groq API (`llama-3.1-8b-instant`)
  - Orchestration: LangChain
  - Embeddings: HuggingFace (`all-MiniLM-L6-v2`)
- **Databases**:
  - Relational: PostgreSQL (Supabase) via SQLAlchemy
  - Vector: ChromaDB

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- A PostgreSQL instance
- Groq API Key (for LLM inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd SmartOps
   ```

2. **Set up environment variables:**
   Copy the example environment file and fill in your credentials.
   ```bash
   cp .env.example .env
   ```
   *Make sure to configure your `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_NAME`.*

3. **Run with Docker (Recommended):**
   ```bash
   docker-compose up --build
   ```
   The backend API will be available at `http://localhost:8000`.

### Local Development Setup

If you prefer to run it locally without Docker:

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

4. **Frontend:** Simply open `frontend/index.html` in your browser, or use a local dev server (like Live Server).

## Architecture Overview

When a user submits a query, it flows through our pipeline:
1. **Ingestion**: The query hits the `/chat` endpoint.
2. **Routing**: An LLM extracts intent and target entities.
3. **Planning**: The system dynamically generates an execution plan.
4. **Execution**: Planned steps run sequentially or in parallel by routing to specialized agents.
5. **Generation**: Raw data is synthesized into a natural language response.

## Data Preparation and Notebooks

The project includes a `notebooks` directory, which is used for data exploration and feature engineering.
- `test.ipynb`: Processes raw e-commerce data and generates refined datasets for sales forecasting, customer churn prediction, and sentiment analysis.
- `datasets_prep.py`: Contains helper functions to build these datasets.

## Deployment

- **Backend**: Hosted on Google Cloud Run for serverless container execution.
- **Frontend**: Deployed on Vercel for fast edge delivery.
- **Database**: Cloud PostgreSQL managed via Supabase.
- **CI/CD**: Fully automated pipelines using GitHub Actions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
