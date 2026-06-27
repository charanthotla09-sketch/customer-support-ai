import os
import re
import json
import sqlite3
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please add it to your .env file.")

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# =========================================================
# TASK 7: SQLITE MEMORY
# =========================================================
class SQLiteMemory:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._connect()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            customer_name TEXT,
            updated_at TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            customer_name TEXT,
            query TEXT,
            intent TEXT,
            department TEXT,
            approval_status TEXT,
            response TEXT,
            created_at TEXT
        )
        """)

        conn.commit()
        conn.close()

    def save_customer_name(self, customer_id: str, customer_name: str):
        if not customer_name:
            return

        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO customers (customer_id, customer_name, updated_at)
        VALUES (?, ?, ?)
        """, (customer_id, customer_name, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_customer_name(self, customer_id: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT customer_name FROM customers WHERE customer_id = ?", (customer_id,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    def save_interaction(
        self,
        customer_id: str,
        customer_name: str,
        query: str,
        intent: str,
        department: str,
        approval_status: str,
        response: str
    ):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO interactions
        (customer_id, customer_name, query, intent, department, approval_status, response, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            customer_id,
            customer_name,
            query,
            intent,
            department,
            approval_status,
            response,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def get_recent_history(self, customer_id: str, limit: int = 5):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
        SELECT query, intent, department, approval_status, created_at
        FROM interactions
        WHERE customer_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (customer_id, limit))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return "No previous conversation history found."

        rows = list(reversed(rows))
        formatted = []
        for q, intent, dept, approval, created_at in rows:
            formatted.append(
                f"- [{created_at}] Query: {q} | Intent: {intent} | Department: {dept} | Approval: {approval}"
            )
        return "\n".join(formatted)

    def get_previous_issue(self, customer_id: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
        SELECT query, intent, department, created_at
        FROM interactions
        WHERE customer_id = ?
          AND intent != 'Memory'
        ORDER BY id DESC
        LIMIT 1
        """, (customer_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        query, intent, department, created_at = row
        return f"Your previous support issue was: '{query}' (Intent: {intent}, Department: {department}, Time: {created_at})."


memory_store = SQLiteMemory("memory.db")

# =========================================================
# TASK 6: RAG DOCUMENTS + SIMPLE RETRIEVER
# =========================================================
class SimpleRAG:
    def __init__(self, docs_dir="docs"):
        self.docs_dir = docs_dir
        self.documents = []
        self.vectorizer = None
        self.matrix = None

        self._ensure_docs()
        self._load_and_index_docs()

    def _ensure_docs(self):
        os.makedirs(self.docs_dir, exist_ok=True)

        docs_content = {
            "company_policy.txt": """
Company Policy Document

1. Refund requests must be reviewed and approved by a human supervisor before confirmation is sent.
2. Subscription cancellation requests must be reviewed by a human supervisor.
3. Account closure requests must not be auto-approved.
4. Compensation requests require management approval.
5. Escalation to management must be reviewed by a human supervisor.
6. Customer communication must remain respectful, clear, and professional.
            """.strip(),

            "pricing_guide.txt": """
Pricing Guide

1. Basic Plan: $10/month
   - Core business management features
   - Email support

2. Pro Plan: $30/month
   - Advanced analytics
   - Team collaboration tools
   - Priority email support

3. Enterprise Plan: Custom pricing
   - Dedicated account manager
   - Advanced security controls
   - Custom integrations

4. Annual subscriptions receive a 20% discount.
            """.strip(),

            "technical_manual.txt": """
Technical Manual

1. If the application crashes during file upload:
   - Clear the browser/application cache
   - Confirm the file size is under 50 MB
   - Update the application to the latest version
   - Retry using a supported file format
   - Review system logs for upload-related errors

2. Login problems:
   - Verify username and password
   - Reset password if needed
   - Check whether the account is active

3. Installation issues:
   - Verify system requirements
   - Reinstall the application
   - Check network/firewall restrictions

4. Configuration issues:
   - Use the default configuration template
   - Validate API keys and environment settings
            """.strip(),

            "faq_document.txt": """
FAQ Document

1. Password Reset:
   Click "Forgot Password" on the login page and follow the reset instructions.

2. Refund Processing:
   Approved refunds are generally processed within 5-7 business days.

3. Invoice Requests:
   Customers can request invoices from the billing department.

4. Profile Updates:
   Customers can update profile information from the account settings page.

5. Account Activation/Deactivation:
   Contact account support if activation or deactivation assistance is required.
            """.strip()
        }

        for filename, content in docs_content.items():
            path = os.path.join(self.docs_dir, filename)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

    def _chunk_text(self, text, chunk_size=400, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def _load_and_index_docs(self):
        self.documents = []

        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".txt"):
                path = os.path.join(self.docs_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = self._chunk_text(text)
                for chunk in chunks:
                    self.documents.append({
                        "source": filename,
                        "content": chunk
                    })

        corpus = [doc["content"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        ranked_indices = scores.argsort()[::-1]
        selected = []

        for idx in ranked_indices:
            if len(selected) >= top_k:
                break
            if scores[idx] > 0:
                selected.append(self.documents[idx])

        if not selected:
            selected = self.documents[:top_k]

        sources = sorted(set(doc["source"] for doc in selected))
        context = "\n\n".join(
            [f"[Source: {doc['source']}]\n{doc['content']}" for doc in selected]
        )
        return context, sources


rag = SimpleRAG()

# =========================================================
# TASK 2: STATE STRUCTURE
# =========================================================
class SupportState(TypedDict, total=False):
    customer_id: str
    customer_name: str
    query: str

    intent: str
    department: str
    requires_approval: bool
    approval_status: str

    retrieved_context: str
    retrieved_sources: str

    recalled_history: str
    conversation_history: str

    final_response: str

# =========================================================
# HELPERS
# =========================================================
def extract_name(query: str):
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z'-]*)",
        r"\bi am\s+([A-Za-z][A-Za-z'-]*)",
        r"\bi'm\s+([A-Za-z][A-Za-z'-]*)"
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def parse_json_from_text(text: str):
    try:
        text = text.strip().replace("```json", "").replace("```", "")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None

def heuristic_classification(query: str):
    q = query.lower()

    # Memory intent
    memory_phrases = [
        "what was my previous issue",
        "what was my previous support issue",
        "previous support issue",
        "previous issue",
        "past issue",
        "chat history",
        "history"
    ]
    if any(p in q for p in memory_phrases):
        return {
            "intent": "Memory",
            "department": "None",
            "requires_approval": False
        }

    # High-risk detection
    high_risk_phrases = [
        "refund",
        "cancel subscription",
        "subscription cancellation",
        "close my account",
        "account closure",
        "compensation",
        "escalate",
        "escalation to management",
        "management"
    ]
    requires_approval = any(p in q for p in high_risk_phrases)

    # Billing
    billing_keywords = [
        "refund", "invoice", "billing", "payment", "charged", "charge",
        "subscription", "cancel", "receipt"
    ]
    if any(k in q for k in billing_keywords):
        return {
            "intent": "Billing",
            "department": "Billing",
            "requires_approval": requires_approval
        }

    # Account
    account_keywords = [
        "password", "profile", "account", "activate", "deactivate",
        "reset password", "forgot password"
    ]
    if any(k in q for k in account_keywords):
        return {
            "intent": "Account",
            "department": "Account",
            "requires_approval": requires_approval
        }

    # Technical
    technical_keywords = [
        "crash", "error", "bug", "upload", "installation", "install",
        "login problem", "configuration", "config", "not working", "issue"
    ]
    if any(k in q for k in technical_keywords):
        return {
            "intent": "Technical",
            "department": "Technical Support",
            "requires_approval": requires_approval
        }

    # Sales
    sales_keywords = [
        "pricing", "price", "plans", "plan", "software", "product",
        "subscription plans", "features", "cost"
    ]
    if any(k in q for k in sales_keywords):
        return {
            "intent": "Sales",
            "department": "Sales",
            "requires_approval": requires_approval
        }

    return None

def llm_classification(query: str):
    prompt = f"""
You are an intent classifier for a SaaS customer support system.

Classify the user query into one of these intents only:
- Sales
- Technical
- Billing
- Account
- Memory

Also decide whether it requires human approval.
Human approval is required for:
- Refund requests
- Subscription cancellation
- Account closure requests
- Compensation requests
- Escalation to management

Return ONLY valid JSON in this format:
{{
  "intent": "Sales|Technical|Billing|Account|Memory",
  "department": "Sales|Technical Support|Billing|Account|None",
  "requires_approval": true_or_false
}}

User query:
{query}
    """.strip()

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = parse_json_from_text(response.content)
        if data:
            return data
    except Exception:
        pass

    return {
        "intent": "Sales",
        "department": "Sales",
        "requires_approval": False
    }

def fallback_response(state: SupportState):
    customer_name = state.get("customer_name") or "Customer"

    if state.get("intent") == "Memory":
        recalled = state.get("recalled_history") or "No previous issue was found."
        return f"{customer_name}, {recalled}"

    if state.get("approval_status") == "Rejected":
        return (
            f"{customer_name}, your request cannot be automatically completed at this time "
            f"because it was not approved by the human supervisor."
        )

    context = state.get("retrieved_context", "No context found.")
    return (
        f"{customer_name}, here is the support information I found:\n\n{context}\n\n"
        f"If you need more help, please let us know."
    )

# =========================================================
# TASK 3: INTENT CLASSIFIER NODE
# =========================================================
def classifier_node(state: SupportState):
    print("\n[Classifier] Analyzing customer query...")

    customer_id = state["customer_id"]
    query = state["query"]

    existing_name = memory_store.get_customer_name(customer_id)
    found_name = extract_name(query)
    customer_name = found_name or existing_name or ""

    result = heuristic_classification(query)
    if not result:
        result = llm_classification(query)

    intent = result["intent"]
    department = result["department"]
    requires_approval = result["requires_approval"]
    approval_status = "Pending" if requires_approval else "Not Required"

    print(f"[Classifier] Intent: {intent}")
    print(f"[Classifier] Department: {department}")
    print(f"[Classifier] Requires Approval: {requires_approval}")

    return {
        "customer_name": customer_name,
        "intent": intent,
        "department": department,
        "requires_approval": requires_approval,
        "approval_status": approval_status
    }

# =========================================================
# TASK 4: CONDITIONAL ROUTING
# =========================================================
def route_after_classification(state: SupportState):
    if state["intent"] == "Memory":
        return "memory_recall"

    if state["requires_approval"]:
        return "human_approval"

    dept = state.get("department", "")
    if dept == "Sales":
        return "sales_agent"
    if dept == "Technical Support":
        return "technical_agent"
    if dept == "Billing":
        return "billing_agent"
    if dept == "Account":
        return "account_agent"

    return "supervisor"

# =========================================================
# TASK 5: SPECIALIZED AGENTS
# =========================================================
def run_department_agent(state: SupportState, department: str):
    print(f"\n[{department}] Retrieving knowledge base information...")

    query = state["query"]
    context, sources = rag.retrieve(f"{department}: {query}", top_k=3)

    print(f"[RAG] Sources Retrieved: {', '.join(sources)}")

    return {
        "retrieved_context": context,
        "retrieved_sources": ", ".join(sources)
    }

def sales_agent(state: SupportState):
    return run_department_agent(state, "Sales")

def technical_agent(state: SupportState):
    return run_department_agent(state, "Technical Support")

def billing_agent(state: SupportState):
    return run_department_agent(state, "Billing")

def account_agent(state: SupportState):
    return run_department_agent(state, "Account")

# =========================================================
# TASK 8: HUMAN-IN-THE-LOOP
# =========================================================
def human_approval_node(state: SupportState):
    print("\n[HITL] High-risk request detected.")
    print(f"[HITL] Query: {state['query']}")
    print("[HITL] This request requires human supervisor approval.")

    while True:
        decision = input("Supervisor decision (Approve/Reject): ").strip().lower()
        if decision in ["approve", "approved"]:
            print("[HITL] Request approved by supervisor.")
            return {"approval_status": "Approved"}
        elif decision in ["reject", "rejected"]:
            print("[HITL] Request rejected by supervisor.")
            return {"approval_status": "Rejected"}
        else:
            print("Please type only: Approve or Reject")

def route_after_approval(state: SupportState):
    if state["approval_status"] == "Rejected":
        return "supervisor"

    dept = state.get("department", "")
    if dept == "Sales":
        return "sales_agent"
    if dept == "Technical Support":
        return "technical_agent"
    if dept == "Billing":
        return "billing_agent"
    if dept == "Account":
        return "account_agent"

    return "supervisor"

# =========================================================
# MEMORY RECALL NODE
# =========================================================
def memory_recall_node(state: SupportState):
    print("\n[Memory] Retrieving previous customer history from SQLite...")

    customer_id = state["customer_id"]
    previous_issue = memory_store.get_previous_issue(customer_id)
    conversation_history = memory_store.get_recent_history(customer_id, limit=5)

    if not previous_issue:
        previous_issue = "No previous support issue was found for this customer."

    print(f"[Memory] Previous Issue: {previous_issue}")

    return {
        "recalled_history": previous_issue,
        "conversation_history": conversation_history
    }

# =========================================================
# TASK 9: SUPERVISOR AGENT
# =========================================================
def supervisor_node(state: SupportState):
    print("\n[Supervisor] Validating and preparing final response...")

    customer_id = state["customer_id"]
    customer_name = state.get("customer_name") or memory_store.get_customer_name(customer_id) or "Customer"
    query = state["query"]

    conversation_history = memory_store.get_recent_history(customer_id, limit=3)
    retrieved_context = state.get("retrieved_context", "No retrieved document context.")
    recalled_history = state.get("recalled_history", "")
    approval_status = state.get("approval_status", "Not Required")
    department = state.get("department", "None")
    intent = state.get("intent", "Unknown")

    system_prompt = f"""
You are the Supervisor Agent for ABC Technologies customer support.

Your job:
1. Review the customer query.
2. Use the retrieved company document context.
3. Use customer memory/history when relevant.
4. If approval_status is Rejected, politely decline the request.
5. If the query asks about previous issues, answer from memory/history directly.
6. Keep the answer professional, helpful, and concise.

Customer Name: {customer_name}
Customer Query: {query}
Intent: {intent}
Department: {department}
Approval Status: {approval_status}

Retrieved Context:
{retrieved_context}

Previous Issue:
{recalled_history}

Recent Conversation History:
{conversation_history}
    """.strip()

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate the final customer response.")
        ])
        final_response = response.content.strip()
    except Exception as e:
        print(f"[Supervisor] Gemini failed, using fallback response. Error: {e}")
        final_response = fallback_response(state)

    print("[Supervisor] Final response generated.")

    return {
        "conversation_history": conversation_history,
        "final_response": final_response
    }

# =========================================================
# SAVE MEMORY NODE
# =========================================================
def save_memory_node(state: SupportState):
    print("\n[Memory] Saving interaction to SQLite...")

    customer_id = state["customer_id"]
    customer_name = state.get("customer_name", "")
    query = state["query"]
    intent = state.get("intent", "")
    department = state.get("department", "")
    approval_status = state.get("approval_status", "")
    final_response = state.get("final_response", "")

    if customer_name:
        memory_store.save_customer_name(customer_id, customer_name)

    memory_store.save_interaction(
        customer_id=customer_id,
        customer_name=customer_name,
        query=query,
        intent=intent,
        department=department,
        approval_status=approval_status,
        response=final_response
    )

    print("[Memory] Interaction saved successfully.")
    return {}

# =========================================================
# TASK 1: LANGGRAPH WORKFLOW
# =========================================================
builder = StateGraph(SupportState)

builder.add_node("classifier", classifier_node)
builder.add_node("sales_agent", sales_agent)
builder.add_node("technical_agent", technical_agent)
builder.add_node("billing_agent", billing_agent)
builder.add_node("account_agent", account_agent)
builder.add_node("human_approval", human_approval_node)
builder.add_node("memory_recall", memory_recall_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("save_memory", save_memory_node)

builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    route_after_classification,
    {
        "sales_agent": "sales_agent",
        "technical_agent": "technical_agent",
        "billing_agent": "billing_agent",
        "account_agent": "account_agent",
        "human_approval": "human_approval",
        "memory_recall": "memory_recall",
        "supervisor": "supervisor"
    }
)

builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "sales_agent": "sales_agent",
        "technical_agent": "technical_agent",
        "billing_agent": "billing_agent",
        "account_agent": "account_agent",
        "supervisor": "supervisor"
    }
)

builder.add_edge("sales_agent", "supervisor")
builder.add_edge("technical_agent", "supervisor")
builder.add_edge("billing_agent", "supervisor")
builder.add_edge("account_agent", "supervisor")
builder.add_edge("memory_recall", "supervisor")

builder.add_edge("supervisor", "save_memory")
builder.add_edge("save_memory", END)

app = builder.compile()

# =========================================================
# TASK 10: DEMONSTRATION
# =========================================================
def run_single_query(customer_id: str, query: str):
    state = {
        "customer_id": customer_id,
        "query": query
    }

    result = app.invoke(state)

    print("\n" + "=" * 70)
    print("FINAL RESPONSE")
    print("=" * 70)
    print(result["final_response"])
    print("=" * 70 + "\n")

def run_demo():
    print("\n" + "=" * 70)
    print("ABC TECHNOLOGIES - AI CUSTOMER SUPPORT AUTOMATION DEMO")
    print("=" * 70)

    customer_id = "customer_david_001"

    demo_queries = [
        "What are the pricing plans available for your software?",
        "I forgot my account password.",
        "My application crashes whenever I upload a file.",
        "My name is David. I need a refund for my annual subscription.",
        "What was my previous support issue?"
    ]

    for i, query in enumerate(demo_queries, start=1):
        print(f"\n\n--- QUERY {i} ---")
        print(f"Customer: {query}")
        run_single_query(customer_id, query)

def interactive_chat():
    print("\nInteractive mode started.")
    customer_id = input("Enter customer ID: ").strip()

    while True:
        query = input("\nEnter customer query (or type 'exit'): ").strip()
        if query.lower() == "exit":
            print("Exiting chat.")
            break

        run_single_query(customer_id, query)

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run Demo")
    print("2. Interactive Chat")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_demo()
    elif choice == "2":
        interactive_chat()
    else:
        print("Invalid choice. Running demo by default.")
        run_demo()