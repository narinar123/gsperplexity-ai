import os
import re
import httpx
import streamlit as st
from urllib.parse import urlparse

# ── Secret loading: works on both local (.env) and Streamlit Cloud (st.secrets) ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed on Streamlit Cloud

def get_secret(key: str) -> str:
    """Read from st.secrets (cloud) first, fallback to os.environ (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv(key, "")

# LangChain core
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Optional: Google Gemini (uses GOOGLE_API_KEY)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Optional: Tavily search
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Optional: trafilatura for URL content extraction (lightweight, no BS4 needed)
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

# Optional: DuckDuckGo fallback search (free, no API key)
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="gsperplexity.ai - AI Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — Glassmorphism, animations, typography
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import premium typography */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Apply custom theme variables */
:root {
    --bg-primary: #090a0f;
    --bg-secondary: rgba(22, 28, 45, 0.4);
    --border-color: rgba(255, 255, 255, 0.08);
    --accent-purple: #8b5cf6;
    --accent-blue: #3b82f6;
    --accent-glow: rgba(139, 92, 246, 0.15);
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
}

/* Background styling */
.stApp {
    background: linear-gradient(180deg, #090a0f 0%, #0c0f1d 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main) !important;
}

/* Typography styles */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Logo and Header Styling */
.logo-container {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    margin-bottom: 1rem;
    animation: fadeInDown 0.8s ease-out;
}
.logo-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
    filter: drop-shadow(0px 2px 10px rgba(139, 92, 246, 0.25));
}
.logo-subtitle {
    font-size: 1.15rem;
    color: var(--text-muted);
    font-weight: 400;
}

/* Glassmorphic Cards for Sources */
.sources-header {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem 0;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sources-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.source-card {
    background: var(--bg-secondary);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    text-decoration: none !important;
    color: var(--text-main) !important;
    display: flex;
    flex-direction: column;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.source-card:hover {
    transform: translateY(-4px);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 12px 24px -10px rgba(139, 92, 246, 0.3);
    background: rgba(30, 41, 59, 0.65);
}
.source-title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.source-favicon {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    background-color: rgba(255, 255, 255, 0.1);
}
.source-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-grow: 1;
}
.source-index {
    background: rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 6px;
}
.source-snippet {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    flex-grow: 1;
}
.source-url {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.6rem;
    word-break: break-all;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Beautiful Interactive Citation Pills */
.citation-pill {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(139, 92, 246, 0.18) !important;
    color: #a78bfa !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    padding: 0px 6px !important;
    border-radius: 8px !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-decoration: none !important;
    margin: 0 2px !important;
    vertical-align: super !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.citation-pill:hover {
    background: #8b5cf6 !important;
    color: #ffffff !important;
    transform: scale(1.1) translateY(-1px) !important;
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.5) !important;
}

/* Tool badge — shows which agent tool was used */
.tool-badge {
    background: rgba(59, 130, 246, 0.1);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.25);
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    animation: fadeIn 0.4s ease-out;
}

/* Query Badge Styling */
.query-badge {
    background: rgba(59, 130, 246, 0.1);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.25);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    animation: fadeIn 0.5s ease-out;
}

/* Status container */
.status-container {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-color: #0b0c10 !important;
    border-right: 1px solid var(--border-color) !important;
}

/* Chat message bubbles */
.stChatMessage {
    background: rgba(22, 28, 45, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    margin-bottom: 0.75rem !important;
}

/* Custom animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_system_rules() -> str:
    """Load AI behavior rules from markdown.md."""
    try:
        with open("markdown.md", "r") as f:
            return f.read()
    except Exception:
        return "You are an AI research assistant. Provide accurate, well-cited answers using your tools."


def format_citations(text: str, sources: list) -> str:
    """Convert [N] or [web:N] markers into interactive HTML citation pills."""
    def replace_citation(match):
        num = int(match.group(1))
        if 1 <= num <= len(sources):
            src = sources[num - 1]
            return f'<a class="citation-pill" href="{src["url"]}" target="_blank">{num}</a>'
        return f'[{num}]'

    text = re.sub(r'\[web:(\d+)\]', replace_citation, text)
    text = re.sub(r'\[(\d+)\]', replace_citation, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOOLS — All cloud API calls, zero local compute
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for current, up-to-date information, news, prices, or facts.
    Always use this tool first for any factual question. Returns numbered results with URLs."""
    # Initialize sources store for this session turn
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []

    tavily_key = get_secret("TAVILY_API_KEY")
    serper_key = get_secret("SERPER_API_KEY")

    # — Primary: Tavily —
    if TAVILY_AVAILABLE and tavily_key:
        try:
            client = TavilyClient(api_key=tavily_key)
            resp = client.search(query=query, max_results=5, search_depth="basic")
            results = resp.get("results", [])
            output_parts = []
            for r in results:
                url = r.get("url", "")
                title = r.get("title", "Untitled")
                content = r.get("content", "")
                # Track unique sources
                existing_urls = {s["url"] for s in st.session_state.current_sources}
                if url and url not in existing_urls:
                    st.session_state.current_sources.append({"title": title, "url": url, "content": content})
                idx = next((i + 1 for i, s in enumerate(st.session_state.current_sources) if s["url"] == url), "?")
                output_parts.append(f"[{idx}] {title}\nURL: {url}\n{content[:600]}")
            if output_parts:
                return "\n\n".join(output_parts)
        except Exception:
            pass  # Fall through to DuckDuckGo

    # — Fallback: DuckDuckGo (free, no key needed) —
    if DDG_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            output_parts = []
            for r in results:
                url = r.get("href", "")
                title = r.get("title", "Untitled")
                content = r.get("body", "")
                existing_urls = {s["url"] for s in st.session_state.current_sources}
                if url and url not in existing_urls:
                    st.session_state.current_sources.append({"title": title, "url": url, "content": content})
                idx = next((i + 1 for i, s in enumerate(st.session_state.current_sources) if s["url"] == url), "?")
                output_parts.append(f"[{idx}] {title}\nURL: {url}\n{content[:600]}")
            if output_parts:
                return "\n\n".join(output_parts)
        except Exception as e:
            return f"Search unavailable: {str(e)}"

    return "No search results found. TAVILY_API_KEY may be missing or DuckDuckGo is unreachable."


@tool
def fetch_url(url: str) -> str:
    """Fetch and extract the full clean text from a specific webpage URL.
    Use this when you need more detail from a specific source beyond what search_web returned."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = httpx.get(url, timeout=12, follow_redirects=True, headers=headers)
        resp.raise_for_status()

        if TRAFILATURA_AVAILABLE:
            content = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            if content:
                return content[:4000]

        # Fallback: return raw text snippet
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:3000]

    except httpx.TimeoutException:
        return f"Timeout fetching {url}. Try a different source."
    except Exception as e:
        return f"Could not fetch URL: {str(e)}"


@tool
def execute_code(task_description: str) -> str:
    """Solve mathematical calculations, unit conversions, logic problems, statistics, or data analysis.
    Describe the problem clearly and the AI will solve it step by step using reasoning.
    Examples: 'calculate compound interest for $5000 at 7% over 10 years',
              'convert 98.6°F to Celsius', 'find the median of [3, 7, 2, 9, 4]'"""
    # Cloud-interpreted: returns structured prompt for the main LLM to reason through
    return (
        f"COMPUTATION TASK: {task_description}\n\n"
        "Please solve this step by step, showing your work clearly. "
        "Use mathematical notation where appropriate and provide the final answer prominently."
    )


@tool
def search_pplx_support(query: str) -> str:
    """Search for Perplexity AI specific help, features, documentation, or support articles.
    Use this when the user asks about Perplexity AI itself, its features, pricing, or how it works."""
    try:
        # Try Tavily with Perplexity domain focus
        tavily_key = get_secret("TAVILY_API_KEY")
        if TAVILY_AVAILABLE and tavily_key:
            client = TavilyClient(api_key=tavily_key)
            resp = client.search(
                query=f"Perplexity AI {query}",
                max_results=4,
                search_depth="basic",
                include_domains=["perplexity.ai", "docs.perplexity.ai", "blog.perplexity.ai"]
            )
            results = resp.get("results", [])
            if results:
                output = [f"[{i+1}] {r['title']}\nURL: {r['url']}\n{r['content'][:500]}" for i, r in enumerate(results)]
                return "\n\n".join(output)

        # Fallback: DuckDuckGo scoped to perplexity.ai
        if DDG_AVAILABLE:
            with DDGS() as ddgs:
                results = list(ddgs.text(f"site:perplexity.ai {query}", max_results=4))
            if results:
                output = [f"[{i+1}] {r.get('title','')}\nURL: {r.get('href','')}\n{r.get('body','')[:500]}" for i, r in enumerate(results)]
                return "\n\n".join(output)

        return "Could not find Perplexity support information."
    except Exception as e:
        return f"Support search failed: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [search_web, fetch_url, execute_code, search_pplx_support]

TOOL_ICONS = {
    "search_web": "🔍",
    "fetch_url": "🌐",
    "execute_code": "🧮",
    "search_pplx_support": "💬",
}


def build_agent(llm, system_rules: str) -> AgentExecutor:
    """Create a LangChain Tool Calling Agent with all 4 tools."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOGO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo-container">
    <div class="logo-title">gsperplexity.ai</div>
    <div class="logo-subtitle">Intelligent AI Research Agent · Powered by LangChain, Tavily & Cloud LLMs</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Engine Settings")

openrouter_key = get_secret("OPENROUTER_API_KEY")
tavily_key     = get_secret("TAVILY_API_KEY")
google_key     = get_secret("GOOGLE_API_KEY")

if not openrouter_key:
    st.sidebar.error("⚠️ OPENROUTER_API_KEY missing in .env")
if not tavily_key:
    st.sidebar.warning("⚠️ TAVILY_API_KEY missing — using DuckDuckGo fallback")

# Models — now includes Gemini via direct Google API (uses GOOGLE_API_KEY)
MODEL_OPTIONS = {
    "🚀 Gemini 2.5 Flash (Google Direct)":   ("google",      "gemini-2.5-flash-preview-05-20"),
    "⚡ Gemini 2.0 Flash (Google Direct)":    ("google",      "gemini-2.0-flash"),
    "🧠 GPT-4o Mini (OpenRouter)":            ("openrouter",  "openai/gpt-4o-mini"),
    "💜 Claude 3.5 Haiku (OpenRouter)":       ("openrouter",  "anthropic/claude-3.5-haiku"),
    "🔮 Gemini 2.5 Flash (OpenRouter)":       ("openrouter",  "google/gemini-2.5-flash"),
    "🦙 Llama 3.1 8B — Free (OpenRouter)":    ("openrouter",  "meta-llama/llama-3.1-8b-instruct:free"),
}

selected_label = st.sidebar.selectbox("Choose Model", list(MODEL_OPTIONS.keys()))
provider, model_id = MODEL_OPTIONS[selected_label]
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛠️ Active Agent Tools
- 🔍 **search_web** — Tavily + DuckDuckGo fallback
- 🌐 **fetch_url** — Deep URL extraction (trafilatura)
- 🧮 **execute_code** — Math & logic (cloud-interpreted)
- 💬 **search_pplx_support** — Perplexity AI docs
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 How It Works
1. **Agent** autonomously picks the right tools
2. **Tools** fetch live data from the web
3. **LLM** synthesizes a cited answer
4. **History** keeps context across queries
""")

st.sidebar.markdown("---")
# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.current_sources = []
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — Chat history + source tracking
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything…")

if user_input:
    # Validate keys
    if provider == "google" and not google_key:
        st.error("🔑 GOOGLE_API_KEY not found in .env — switch to an OpenRouter model or add the key.")
        st.stop()
    if provider == "openrouter" and not openrouter_key:
        st.error("🔑 OPENROUTER_API_KEY not found in .env — please add it.")
        st.stop()

    # Store & display user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Reset sources for this new query
    st.session_state.current_sources = []

    # ── Build LLM ──────────────────────────────────────────────────────────
    if provider == "google" and GOOGLE_AVAILABLE and google_key:
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            google_api_key=google_key,
        )
    else:
        llm = ChatOpenAI(
            model=model_id,
            temperature=temperature,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    # ── Run Agent ──────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        tool_calls_made = []
        raw_response = ""

        with st.status("🤖 Agent researching…", expanded=True) as status_box:
            try:
                system_rules = load_system_rules()
                agent_executor = build_agent(llm, system_rules)

                # Last 6 messages as history context (keeps it light)
                history_context = st.session_state.messages[:-1][-6:]

                st.write("🧠 Agent initialised — selecting tools…")

                result = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": history_context,
                })

                # Show which tools the agent used
                steps = result.get("intermediate_steps", [])
                if steps:
                    for action, observation in steps:
                        icon = TOOL_ICONS.get(action.tool, "🔧")
                        tool_input_preview = str(action.tool_input)[:80]
                        st.write(f"{icon} `{action.tool}` ← {tool_input_preview}")
                        tool_calls_made.append(action.tool)
                else:
                    st.write("💡 Agent answered from knowledge (no tool call needed)")

                raw_response = result.get("output", "No response generated.")
                status_box.update(label="✅ Research complete!", state="complete", expanded=False)

            except Exception as e:
                raw_response = f"⚠️ Agent encountered an error: `{str(e)}`\n\nTry a different model or rephrase your question."
                status_box.update(label="⚠️ Error occurred", state="error", expanded=False)

        # ── Sources Grid ───────────────────────────────────────────────────
        sources = st.session_state.get("current_sources", [])
        if sources:
            st.markdown('<div class="sources-header">📚 Sources</div>', unsafe_allow_html=True)
            num_cols = min(3, len(sources))
            cols = st.columns(num_cols)
            for idx, source in enumerate(sources):
                col_idx = idx % num_cols
                parsed_url = urlparse(source["url"])
                domain = parsed_url.netloc
                favicon_url = f"https://www.google.com/s2/favicons?sz=64&domain={domain}"
                snippet = source.get("content", "")[:200]

                card_html = f"""
                <a class="source-card" href="{source['url']}" target="_blank">
                    <div class="source-title-row">
                        <img class="source-favicon" src="{favicon_url}" alt="" onerror="this.style.display='none'">
                        <div class="source-title">{source['title']}</div>
                        <span class="source-index">{idx + 1}</span>
                    </div>
                    <div class="source-snippet">{snippet}</div>
                    <div class="source-url">{domain}</div>
                </a>
                """
                cols[col_idx].markdown(card_html, unsafe_allow_html=True)

        # ── Formatted Answer ───────────────────────────────────────────────
        st.markdown('<div class="sources-header">✍️ Answer</div>', unsafe_allow_html=True)
        formatted_response = format_citations(raw_response, sources)
        st.markdown(formatted_response, unsafe_allow_html=True)

        # ── Tool usage summary badge row ───────────────────────────────────
        if tool_calls_made:
            unique_tools = list(dict.fromkeys(tool_calls_made))
            badges_html = "".join(
                f'<span class="tool-badge">{TOOL_ICONS.get(t, "🔧")} {t}</span>'
                for t in unique_tools
            )
            st.markdown(f"<div style='margin-top:0.75rem'>{badges_html}</div>", unsafe_allow_html=True)

    # Store assistant message in history
    st.session_state.messages.append(AIMessage(content=raw_response))
