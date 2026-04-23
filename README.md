# 🎬 AutoStream Conversational AI Agent (Gemini Version)

An agentic, LangGraph-powered sales assistant for AutoStream using **Google Gemini 1.5 Flash** (free tier).

## 📁 Project Structure

```
autostream-gemini/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph state machine & all nodes
│   └── rag.py            # RAG pipeline
├── tools/
│   ├── __init__.py
│   └── lead_capture.py   # Mock lead capture function
├── knowledge_base/
│   └── autostream_kb.json
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ How to Run Locally

### 1. Get a free Gemini API key
Go to 👉 https://aistudio.google.com/apikey → Sign in → Create API Key

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Create your .env file
```bash
python -c "open('.env', 'w', encoding='utf-8').write('GOOGLE_API_KEY=AIza-your-key-here\n')"
```

### 4. Run the agent
```bash
python main.py
```

## 💬 Sample Conversation to Test All Features

```
You: Hi, what plans do you offer?
You: What is included in the Pro plan?
You: Do you offer refunds?
You: I want to sign up for the Pro plan
You: John Smith
You: john@gmail.com
You: YouTube
```
