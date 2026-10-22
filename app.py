import html
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain

st.set_page_config(page_title="Meeting assistant", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1080px; }
h1, h2, h3, .display { font-family: 'Fraunces', Georgia, serif; letter-spacing: -0.01em; }
.brand { font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; font-weight: 600; margin-bottom: .2rem; }
.brand-sub { color: #8D97B0; font-size: .9rem; margin-bottom: 1.2rem; }
.display { font-size: 2.7rem; line-height: 1.12; font-weight: 600; margin: .6rem 0 .5rem 0; }
.lede { font-size: 1.1rem; max-width: 58ch; color: #B7BFD3; line-height: 1.6; }
.wave { margin: 0 0 .4rem 0; opacity: .95; }
.meta { color: #8D97B0; margin-bottom: .2rem; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.1rem 0 1.3rem 0; }
.tile { border: 1px solid #263049; background: #151B2D; border-radius: 10px; padding: 14px 16px; }
.tile .n { font-family: 'Fraunces', Georgia, serif; font-size: 1.9rem; line-height: 1.1; }
.tile .l { color: #8D97B0; font-size: .85rem; margin-top: 2px; }
@media (max-width: 700px) { .tiles { grid-template-columns: repeat(2, 1fr); } .display { font-size: 2rem; } }
</style>
""",
    unsafe_allow_html=True,
)

LANGS = {
    "English": "english",
    "Hindi or Hinglish (translated to English)": "hinglish",
}
FILE_TYPES = ["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"]
SUGGESTIONS = [
    "What were the main points?",
    "What was decided?",
    "What are the next steps?",
    "Who owns which task?",
]

st.session_state.setdefault("result", None)
st.session_state.setdefault("chat", [])


# ---------- small helpers ----------
def waveform(n=64, h=64):
    bars = []
    step = 600 / n
    for i in range(n):
        v = math.sin(i * 0.55) + 0.7 * math.sin(i * 0.23 + 1.3) + 0.3 * math.sin(i * 1.1)
        height = 8 + min(abs(v) / 2, 1) * (h - 12)
        op = 0.35 + 0.65 * abs(math.sin(i * 0.17))
        bars.append(
            f'<rect x="{i * step + 1:.1f}" y="{(h - height) / 2:.1f}" width="{step - 3:.1f}" '
            f'height="{height:.1f}" rx="2" fill="#7C8CFF" opacity="{op:.2f}"/>'
        )
    return (
        f'<div class="wave"><svg viewBox="0 0 600 {h}" preserveAspectRatio="none" width="100%" '
        f'height="{h}" xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg></div>'
    )


def tiles(items):
    cells = "".join(f'<div class="tile"><div class="n">{v}</div><div class="l">{l}</div></div>' for l, v in items)
    return f'<div class="tiles">{cells}</div>'


def count_items(text):
    return len(re.findall(r"^\s*\d+[\.\)]", text, flags=re.M))


def queue(q):
    st.session_state.queued = q


def reset():
    st.session_state.result = None
    st.session_state.chat = []


# ---------- export ----------
def sections(r, include_transcript):
    s = [
        ("Summary", r["summary"]),
        ("Action items", r["action_items"]),
        ("Key decisions", r["key_decisions"]),
        ("Open questions", r["open_questions"]),
    ]
    if include_transcript:
        s.append(("Full transcript", r["transcript"]))
    return s


def build_txt(r, include_transcript):
    lines = [r["title"], "=" * len(r["title"]), f"Created: {r['created']}", ""]
    for heading, body in sections(r, include_transcript):
        lines += [heading, "-" * len(heading), body.strip(), ""]
    return "\n".join(lines)


def _pdf_safe(text):
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    for old, new in {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": "-", "…": "...", "\u00a0": " "}.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(r, include_transcript):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 9, _pdf_safe(r["title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Created {r['created']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    for heading, body in sections(r, include_transcript):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(230, 233, 250)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 5.5, _pdf_safe(body), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    return bytes(pdf.output())


def safe_name(title):
    return re.sub(r"[^\w\- ]", "", title)[:50].strip().replace(" ", "_") or "meeting"


# ---------- pipeline ----------
def analyze(source, language):
    st.session_state.result = None
    st.session_state.chat = []
    with st.status("Working on your recording", expanded=True) as status:
        try:
            st.write("Preparing the audio...")
            chunks = process_input(source)
            st.write(f"Transcribing {len(chunks)} audio chunk(s). This is the slowest step.")
            transcript = transcribe_all(chunks, language)
            if not transcript.strip():
                raise ValueError("No speech was found in this recording.")
            st.write("Writing the notes and indexing the transcript...")
            with ThreadPoolExecutor(max_workers=4) as ex:
                f_title = ex.submit(generate_title, transcript)
                f_sum = ex.submit(summarize, transcript)
                f_act = ex.submit(extract_action_items, transcript)
                f_dec = ex.submit(extract_key_decisions, transcript)
                f_q = ex.submit(extract_questions, transcript)
                f_rag = ex.submit(build_rag_chain, transcript)
                title = f_title.result().strip().strip('"')
                summary = f_sum.result()
                actions = f_act.result()
                decisions = f_dec.result()
                questions = f_q.result()
                rag_chain = f_rag.result()
        except Exception as e:
            status.update(label="Something went wrong", state="error", expanded=True)
            st.error(f"{type(e).__name__}: {e}")
            return
        status.update(label="Done", state="complete", expanded=False)
    st.session_state.result = {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": actions,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "created": datetime.now().strftime("%d %b %Y, %H:%M"),
    }


# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand">Meeting assistant</div><div class="brand-sub">Notes, decisions and answers from any recording.</div>', unsafe_allow_html=True)
    mode = st.radio("Source", ["YouTube link", "Upload a file"], label_visibility="collapsed")
    url, uploaded = "", None
    if mode == "YouTube link":
        url = st.text_input("YouTube link", placeholder="https://youtu.be/...")
    else:
        uploaded = st.file_uploader("Audio or video file", type=FILE_TYPES)
    lang_label = st.selectbox("Spoken language", list(LANGS))
    run = st.button("Analyze recording", type="primary")
    if st.session_state.result:
        st.button("Start over", on_click=reset)

if run:
    source = None
    if mode == "YouTube link":
        if url.strip():
            source = url.strip()
        else:
            st.sidebar.error("Paste a YouTube link first.")
    elif uploaded is None:
        st.sidebar.error("Choose a file to upload first.")
    else:
        os.makedirs("uploads", exist_ok=True)
        source = os.path.join("uploads", uploaded.name)
        with open(source, "wb") as f:
            f.write(uploaded.getbuffer())
    if source:
        analyze(source, LANGS[lang_label])

# ---------- main area ----------
r = st.session_state.result

if r is None:
    st.markdown(waveform(), unsafe_allow_html=True)
    st.markdown('<div class="display">Turn a recording into notes you can question</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">Add a YouTube link or upload a file in the sidebar. You get a summary, '
        "action items, decisions and open questions, and you can ask about anything that was said.</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**1. Add a recording**\n\nA YouTube link, or an audio or video file.")
    with c2.container(border=True):
        st.markdown("**2. Wait for the analysis**\n\nTranscription is the slowest step. Long recordings take a while.")
    with c3.container(border=True):
        st.markdown("**3. Read, ask, download**\n\nBrowse the notes, chat with the transcript, export as PDF or TXT.")
else:
    words = len(r["transcript"].split())
    st.markdown(waveform(n=96, h=36), unsafe_allow_html=True)
    st.markdown(f'<div class="display" style="font-size:2.3rem">{html.escape(r["title"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta">Analyzed {r["created"]}</div>', unsafe_allow_html=True)
    st.markdown(
        tiles(
            [
                ("Words spoken", f"{words:,}"),
                ("Minutes of speech", f"{max(1, round(words / 150))}"),
                ("Action items", count_items(r["action_items"])),
                ("Decisions", count_items(r["key_decisions"])),
            ]
        ),
        unsafe_allow_html=True,
    )

    c_txt, c_pdf, c_opt = st.columns([1, 1, 2])
    include = c_opt.toggle("Include full transcript in downloads", value=False)
    name = safe_name(r["title"])
    c_txt.download_button("Download TXT", build_txt(r, include), file_name=f"{name}.txt", mime="text/plain")
    c_pdf.download_button("Download PDF", build_pdf(r, include), file_name=f"{name}.pdf", mime="application/pdf")

    tabs = st.tabs(["Summary", "Action items", "Decisions", "Open questions", "Transcript", "Ask the meeting"])

    for tab, key in zip(tabs[:4], ["summary", "action_items", "key_decisions", "open_questions"]):
        with tab:
            with st.container(border=True):
                st.markdown(r[key])

    with tabs[4]:
        query = st.text_input("Search the transcript", placeholder="Search for a word or phrase", label_visibility="collapsed")
        if query.strip():
            hits = [s for s in re.split(r"(?<=[.!?])\s+", r["transcript"]) if query.lower() in s.lower()]
            st.caption(f"{len(hits)} matching sentence(s)" + (", showing the first 50" if len(hits) > 50 else ""))
            pat = re.compile(re.escape(query.strip()), re.I)
            for s in hits[:50]:
                st.markdown("- " + pat.sub(lambda m: f"**{m.group(0)}**", s))
        else:
            st.text_area("Full transcript", r["transcript"], height=420, label_visibility="collapsed")

    with tabs[5]:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if not st.session_state.chat:
            st.caption("Ask anything about the meeting, or start with one of these.")
            for col, q in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
                col.button(q, on_click=queue, args=(q,), key=f"sg_{q}")
        typed = st.chat_input("Ask about this meeting")
        prompt = typed or st.session_state.pop("queued", None)
        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching the transcript..."):
                    try:
                        answer = r["rag_chain"].invoke(prompt)
                    except Exception as e:
                        answer = f"The question could not be answered: {type(e).__name__}: {e}"
                st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
        if st.session_state.chat:
            st.button("Clear conversation", on_click=lambda: st.session_state.chat.clear())