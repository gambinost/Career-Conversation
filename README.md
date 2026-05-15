# 🤖 Career Conversation

> An AI agent that represents me to recruiters — so they can learn about my background, ask questions, and reach me, without me sending a single resume.

Deployed on Hugging Face Spaces · Powered by Groq inference · Built in Python

---

## The idea

Resumes are static. This isn't.

**Career Conversation** is a chatbot trained on my LinkedIn profile, work history, projects, and personal context. Recruiters or anyone curious about my background can have a real conversation with me — and it will never make things up.

### What it can do

- Answer questions about my experience, projects, and skills
- Tell you what I'm currently working on
- Draft a contact email to me if you want to reach out
- If it doesn't know the answer to something, it emails me directly with your question and asks for your contact info so I can follow up

### What it won't do

- Hallucinate experience I don't have
- Answer off-topic questions unrelated to my career

---

## Stack

| Layer | Tech |
|---|---|
| Language | Python |
| LLM Inference | Groq API (fast open-source inference) |
| Notifications | Pushover API |
| Deployment | Hugging Face Spaces |
| Context | Custom knowledge base (LinkedIn + manual data) |

---

## How it works

```
User message
    │
    ▼
System prompt (injected career knowledge base)
    │
    ▼
Groq LLM → generates response
    │
    ├─ Normal answer → returned to user
    │
    └─ Unknown / contact request → Pushover notification to me
```

The entire context — my background, projects, skills, and personality — is injected into the system prompt. The model reasons over it and responds accordingly.

---

## Limitations (v1)

This is an early version with known rough edges:

- Pushover free tier limits notifications
- No persistent conversation memory across sessions
- File structure needs a full refactor
- No proper frontend — runs on HuggingFace default UI

> **A full rebuild is planned** — cleaner stack, email-based notifications, proper deployment, and a custom UI. Think of this as the proof of concept.

---

## Why it exists

I built this because I believe your GitHub profile should be a living portfolio, not a link dump. If a recruiter lands here at 2am and has a question — they should be able to get an answer.

---

## Status

🔁 v1 — functional but being rebuilt from scratch with current expertise.

### Live Demo
https://huggingface.co/spaces/Moamenn/Career_conversation1
