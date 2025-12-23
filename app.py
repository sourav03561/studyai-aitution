#!/usr/bin/env python3
"""
Flask backend — PDF → Summary, Flashcards, Quiz, Ask Q&A, Mindmap (Graphviz),
Video Recommendations, Mock Test

- Summaries, flashcards, quiz, Q&A, mock test: Gemini
- Video recommendations: YouTube Data API + Gemini (for key topics / fallback)
- Mindmap: Graphviz → PNG (base64)
- Supabase:
    - study_materials: stores per-user materials + quiz_stats
    - quiz_performance: per-quiz summary rows
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from flask_cors import CORS
from PIL import Image
from google import genai
import fitz  # PyMuPDF
import pytesseract
import requests

# ----------------------- Supabase -----------------------
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    print("✅ Supabase client initialized")
else:
    print("⚠️ SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. DB features disabled.")

# Graphviz (optional)
try:
    import graphviz
except ImportError:
    graphviz = None

# -------------------------------------------------------------------
# FLASK
# -------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 140 * 1024 * 1024  # 140 MB

DEFAULT_DPI = 300
DEFAULT_MIN_CHAR_THRESHOLD = 40
LLM_TRIM_LIMIT = 120_000

# -------------------------------------------------------------------
# PDF extraction
# -------------------------------------------------------------------
def _ocr_page(page: fitz.Page, dpi: int = DEFAULT_DPI, lang: str = "eng") -> str:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=lang).strip()


def extract_text(pdf_path: str, lang: str = "eng") -> str:
    """Hybrid text extraction: embedded text first, fallback to Tesseract OCR."""
    doc = fitz.open(pdf_path)
    pages: List[str] = []
    for page in doc:
        txt = page.get_text("text").strip()
        if len(txt) < DEFAULT_MIN_CHAR_THRESHOLD:
            txt = _ocr_page(page, dpi=DEFAULT_DPI, lang=lang)
        pages.append(txt)
    doc.close()
    return "\n\n".join(pages)


# -------------------------------------------------------------------
# Gemini helpers
# -------------------------------------------------------------------
def trim(text: str, limit: int = LLM_TRIM_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n\n...[truncated]...\n\n" + text[-limit // 2 :]


def robust_json(s: str):
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def generate_study_material(text: str, topic: Optional[str]) -> dict:
    """Ask Gemini to create summary + key_topics + key_points + flashcards + quiz."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {}

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a teaching assistant. Produce STRICT JSON only (no commentary, no markdown fences).

Schema:
{{
  "summary": "string",
  "key_topics": ["string"],
  "key_points": ["string"],
  "flashcards": [{{"front": "string","back": "string"}}],
  "quiz": [
    {{
      "question": "string",
      "options": ["string"],
      "answer": "string",
      "explanation": "string"
    }}
  ]
}}

Rules:
- key_topics: 8–12 concise topic tags.
- key_points: 8–15 short, concrete bullets.
- Avoid duplicates and empty strings.
- Each quiz question: 3–5 options, exactly one correct.
- Difficulty: medium by default.
- Generate 20–30 quiz questions.
- Generate at least 25 flashcards. Prefer 30–50 if content is long.

Topic hint: {topic or "none"}

DOCUMENT:
{trim(text)}
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    data = robust_json(getattr(resp, "text", "") or "") or {}
    return data


# -------------------------------------------------------------------
# YouTube recommendations (YouTube Data API + Gemini key topics)
# -------------------------------------------------------------------
YOUTUBE_API_URL_SEARCH = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_API_URL_VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def parse_iso8601_duration(duration: Optional[str]) -> int:
    if not duration:
        return 0
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not m:
        return 0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def gemini_key_points_from_text(text: str, max_points: int = 5) -> List[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not text.strip():
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
Extract {max_points} short, distinct key learning topics from the following text.
Return them as a plain list, one per line, no numbering, no bullets, no extra commentary.

Text:
{text}
"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    except Exception as e:
        print("Gemini error while extracting key points:", e)
        return []

    raw = (getattr(resp, "text", "") or "").strip()
    if not raw:
        return []

    lines = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]

    seen = set()
    out: List[str] = []
    for ln in lines:
        low = ln.lower()
        if low not in seen:
            seen.add(low)
            out.append(ln)
        if len(out) >= max_points:
            break
    return out


def fetch_youtube_videos(
    query: str, max_results: int = 6, api_key: Optional[str] = None
) -> List[Dict]:
    if not api_key:
        return []

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
        "relevanceLanguage": "en",
    }
    r = requests.get(YOUTUBE_API_URL_SEARCH, params=params, timeout=15)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return []

    video_ids = ",".join(
        [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
    )
    if not video_ids:
        return []

    vparams = {
        "part": "contentDetails,statistics",
        "id": video_ids,
        "key": api_key,
    }
    vr = requests.get(YOUTUBE_API_URL_VIDEOS, params=vparams, timeout=15)
    vr.raise_for_status()
    vmap = {v["id"]: v for v in vr.json().get("items", [])}

    videos: List[Dict] = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        snippet = it.get("snippet", {}) or {}
        details = vmap.get(vid) or {}
        stats = details.get("statistics") or {}
        video_obj: Dict = {
            "videoId": vid,
            "title": snippet.get("title"),
            "channelTitle": snippet.get("channelTitle"),
            "thumbnail": snippet.get("thumbnails", {})
            .get("high", {})
            .get("url")
            or snippet.get("thumbnails", {}).get("default", {}).get("url"),
            "publishedAt": snippet.get("publishedAt"),
            "description": snippet.get("description"),
            "duration": (details.get("contentDetails") or {}).get("duration"),
            "viewCount": stats.get("viewCount"),
            "likeCount": stats.get("likeCount"),
            "statistics": stats,
        }
        videos.append(video_obj)
    return videos


CURATED_VIDEOS: Dict[str, List[Dict]] = {
    "machine learning": [
        {
            "videoId": "GwIo3gDZCVQ",
            "title": "Machine Learning by Andrew Ng (full course)",
            "channelTitle": "Stanford",
            "thumbnail": "https://i.ytimg.com/vi/GwIo3gDZCVQ/hqdefault.jpg",
            "duration": "PT2H30M",
            "viewCount": "1200000",
            "likeCount": "0",
            "statistics": {"likeCount": "0"},
        },
        {
            "videoId": "Gv9_4yMHFhI",
            "title": "Intro to Machine Learning - Crash Course",
            "channelTitle": "CrashCourse",
            "thumbnail": "https://i.ytimg.com/vi/Gv9_4yMHFhI/hqdefault.jpg",
            "duration": "PT12M",
            "viewCount": "350000",
            "likeCount": "0",
            "statistics": {"likeCount": "0"},
        },
    ],
    "neural networks": [
        {
            "videoId": "aircAruvnKk",
            "title": "Neural Networks Explained in 20 Minutes",
            "channelTitle": "AI Simplified",
            "thumbnail": "https://i.ytimg.com/vi/aircAruvnKk/hqdefault.jpg",
            "duration": "PT20M15S",
            "viewCount": "850000",
            "likeCount": "0",
            "statistics": {"likeCount": "0"},
        }
    ],
}


def curated_for_query(query: str, max_results: int = 6) -> List[Dict]:
    if not query:
        out: List[Dict] = []
        for arr in CURATED_VIDEOS.values():
            out.extend(arr)
            if len(out) >= max_results:
                break
        random.shuffle(out)
        return out[:max_results]

    q = query.lower()
    for topic, vids in CURATED_VIDEOS.items():
        if topic in q:
            return vids[:max_results]

    out: List[Dict] = []
    for arr in CURATED_VIDEOS.values():
        out.extend(arr)
        if len(out) >= max_results:
            break
    random.shuffle(out)
    return out[:max_results]


# -------------------------------------------------------------------
# Mindmap: Graphviz → PNG (base64)
# -------------------------------------------------------------------
def build_mindmap(
    summary: str, key_topics: List[str], key_points: List[str]
) -> Dict[str, Any]:
    if graphviz is None:
        return {
            "error": "Graphviz Python package is not installed.",
            "details": "Run 'pip install graphviz' and install Graphviz system binary (dot).",
        }

    root_title = (key_topics[0] if key_topics else summary.split("\n")[0]).strip() or "Mind Map"

    try:
        dot = graphviz.Digraph(comment=root_title, format="png")

        dot.attr(rankdir="LR")
        dot.graph_attr.update(
            dpi="300",
            size="12,7!",
            ratio="compress",
        )

        dot.node(
            "root",
            root_title,
            shape="box",
            style="filled",
            fillcolor="#1D4ED8",
            fontcolor="white",
        )

        topic_ids: List[str] = []
        for i, topic in enumerate(key_topics):
            tid = f"t{i}"
            topic_ids.append(tid)
            dot.node(tid, topic, shape="box", style="filled", fillcolor="#BFDBFE")
            dot.edge("root", tid)

        if key_points:
            for j, point in enumerate(key_points):
                pid = f"p{j}"
                label = point
                dot.node(pid, label, shape="note", style="filled", fillcolor="#EFF6FF")
                parent_id = topic_ids[j % len(topic_ids)] if topic_ids else "root"
                dot.edge(parent_id, pid)

        if not key_topics and not key_points:
            dot.node("empty", "No topics / key points found", shape="note")

        png_bytes = dot.pipe(format="png")
        img = Image.open(io.BytesIO(png_bytes))
        width, height = img.size
        b64 = base64.b64encode(png_bytes).decode("ascii")

        structure = {
            "root": root_title,
            "topics": key_topics,
            "points_by_topic": {},
        }
        if key_topics and key_points:
            for idx, t in enumerate(key_topics):
                structure["points_by_topic"][t] = []
            for idx, p in enumerate(key_points):
                t = key_topics[idx % len(key_topics)]
                structure["points_by_topic"][t].append(p)

        return {
            "topic": root_title,
            "image_base64": b64,
            "mime": "image/png",
            "width": width,
            "height": height,
            "structure": structure,
        }

    except graphviz.backend.ExecutableNotFound as e:
        return {
            "error": "Graphviz 'dot' executable not found.",
            "details": str(e),
        }
    except Exception as e:
        return {
            "error": "Failed to build mindmap.",
            "details": str(e),
        }


# -------------------------------------------------------------------
# MOCK TEST
# -------------------------------------------------------------------
def generate_mock_test(
    text: str, pattern: List[Dict[str, int]], topic: Optional[str] = None
) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}

    client = genai.Client(api_key=api_key)

    cleaned: List[Dict[str, int]] = []
    total_marks = 0
    total_questions = 0
    lines = []

    for p in pattern:
        try:
            m = int(p.get("marks", 0))
            c = int(p.get("count", 0))
        except Exception:
            continue
        if m <= 0 or c <= 0:
            continue
        cleaned.append({"marks": m, "count": c})
        total_marks += m * c
        total_questions += c
        lines.append(f"- {c} questions of {m} marks each")

    if not cleaned:
        return {"error": "Empty or invalid pattern after cleaning"}

    pattern_desc = "\n".join(lines)
    topic_hint = topic or "none"

    prompt = f"""
You are an expert exam paper setter.

Create a MOCK TEST based ONLY on the document content below.

Mark pattern:
{pattern_desc}

Total questions: {total_questions}
Total marks: {total_marks}

Output STRICT JSON only (no markdown, no commentary) with this schema:

{{
  "total_marks": number,
  "sections": [
    {{
      "marks": number,
      "instructions": "string",
      "questions": [
        {{
          "question": "string",
          "marks": number
        }}
      ]
    }}
  ]
}}

Rules:
- For each entry in the pattern (marks = M, count = C), create one section:
   - section.marks = M
   - section.questions.length = C
- Questions must be self-contained and derived from the document.
- No answers or hints.
- Difficulty: undergraduate-level.

Topic hint: {topic_hint}

DOCUMENT:
{trim(text)}
"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    except Exception as e:
        return {"error": "Gemini mock test generation failed", "details": str(e)}

    raw = getattr(resp, "text", "") or ""
    data = robust_json(raw) or {}

    if not isinstance(data, dict):
        return {"error": "Failed to parse mock test JSON from model"}

    if "total_marks" not in data:
        data["total_marks"] = total_marks

    return {
        "total_marks": data.get("total_marks", total_marks),
        "sections": data.get("sections", []),
        "pattern_used": cleaned,
    }


# -------------------------------------------------------------------
# QUIZ SCORING HELPER (NORMAL + REVISION)
# -------------------------------------------------------------------
def compute_quiz_attempt(
    questions: List[Dict[str, Any]],
    answers: Dict[str, Any],
    prev_stats: Optional[Dict[str, Any]] = None,
    active_indices: Optional[List[int]] = None,
    mode: str = "quiz",  # "quiz" or "revision"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    questions      : full quiz from study_materials.quiz
    answers        : {"0": "A", "2": null, ...}
    prev_stats     : existing quiz_stats JSON or {}
    active_indices : which question indices are part of THIS attempt.
                     - normal quiz: [0..n-1] (or None -> all)
                     - revision: e.g. [0,1,2]
    mode           : just a tag ("quiz"/"revision")

    Returns:
      attempt_row  -> data for quiz_performance insert
      new_stats    -> updated quiz_stats JSON for study_materials
    """
    if prev_stats is None:
        prev_stats = {}

    if active_indices is None:
        active_indices = list(range(len(questions)))

    # ---------- 1. Per-attempt counters ----------
    correct = 0
    wrong = 0
    skipped = 0
    details: List[Dict[str, Any]] = []

    for idx in active_indices:
        if idx < 0 or idx >= len(questions):
            continue
        q = questions[idx]
        correct_answer = q.get("answer")
        # answers might have int or str keys
        user_answer = answers.get(str(idx), answers.get(idx, None))

        if user_answer is None:
            skipped += 1
            is_correct = False
        else:
            if user_answer == correct_answer:
                correct += 1
                is_correct = True
            else:
                wrong += 1
                is_correct = False

        details.append(
            {
                "index": idx,
                "question": q.get("question"),
                "options": q.get("options"),
                "correct_answer": correct_answer,
                "selected_answer": user_answer,
                "is_correct": is_correct,
            }
        )

    total_questions = len(active_indices)
    attempted = correct + wrong
    # ✅ accuracy based only on questions that were part of this attempt
    accuracy = correct / attempted if attempted > 0 else 0.0
    score = float(correct)

    # ---------- 2. Build / update quiz_stats ----------
    history = list(prev_stats.get("history", []))
    per_q = dict(prev_stats.get("per_question", {}))

    attempt_no = len(history) + 1

    history_entry = {
        "score": score,
        "wrong": wrong,
        "correct": correct,
        "skipped": skipped,
        "attempted": attempted,
        "attempt_no": attempt_no,
        "total_questions": total_questions,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    history.append(history_entry)

    # 2b. per_question: update only for active_indices
    for idx in active_indices:
        if idx < 0 or idx >= len(questions):
            continue
        q = questions[idx]
        key = str(idx)
        s = per_q.get(key, {"correct": 0, "skipped": 0, "attempts": 0})
        s["attempts"] = s.get("attempts", 0) + 1

        user_answer = answers.get(str(idx), answers.get(idx, None))
        if user_answer is None:
            s["skipped"] = s.get("skipped", 0) + 1
        elif user_answer == q.get("answer"):
            s["correct"] = s.get("correct", 0) + 1

        per_q[key] = s

    # 2c. last_unsolved = questions tried in this attempt and still never correct overall
    new_last_unsolved: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions):
        key = str(idx)
        s = per_q.get(key, {})
        attempts = s.get("attempts", 0)
        correct_count = s.get("correct", 0)
        if attempts > 0 and correct_count == 0:
            new_last_unsolved.append(
                {
                    "index": idx,
                    "question": q.get("question"),
                    "options": q.get("options"),
                    "correct_answer": q.get("answer"),
                    "selected_answer": None,
                }
            )

    new_quiz_stats = {
        "history": history,
        "per_question": per_q,
        "last_unsolved": new_last_unsolved,
    }

    attempt_row = {
        "correct_answers": correct,
        "wrong_answers": wrong,
        "skipped": skipped,
        "total_questions": total_questions,
        "score": score,
        "accuracy": accuracy,
        "attempt_no": attempt_no,
        "answers": answers,
        "details": details,
        "mode": mode,
    }

    return attempt_row, new_quiz_stats


# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.post("/api/study_material")
def study_material():
    """
    Input:
      multipart/form-data:
        - file: PDF
        or
        - text: raw text
        + optional: topic
        + optional: user_id (Supabase auth user id)

    If user_id is provided AND Supabase is configured:
      - insert a row in study_materials and return material_id
    """
    user_id: Optional[str] = None

    if request.form:
        user_id = request.form.get("user_id")

    if not user_id:
        data_json = request.get_json(silent=True) or {}
        if isinstance(data_json, dict):
            user_id = data_json.get("user_id")

    pdf_file = request.files.get("file")
    incoming_text = request.form.get("text") if request.form else None
    topic = request.form.get("topic") if request.form else None

    if not incoming_text:
        data_json = request.get_json(silent=True) or {}
        if isinstance(data_json, dict):
            incoming_text = data_json.get("text")
            if not topic:
                topic = data_json.get("topic")

    if not pdf_file and not incoming_text:
        return abort(400, "Provide file or text")

    if pdf_file:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            pdf_file.save(tmp.name)
            text = extract_text(tmp.name)
        source_type = "pdf"
        source_name = pdf_file.filename
    else:
        text = incoming_text or ""
        source_type = "text"
        source_name = None

    notes = None
    data: Dict[str, Any] = {}
    try:
        data = generate_study_material(text, topic)
        print("Generated study material:", data)
    except Exception as e:
        print("Gemini study_material failed:", e)
        notes = str(e)

    summary = data.get("summary", "")
    key_topics = data.get("key_topics", []) or []
    key_points = data.get("key_points", []) or []
    flashcards = data.get("flashcards", []) or []
    quiz = data.get("quiz", []) or []
    material_id: Optional[str] = None

    # Supabase storage (only if configured AND user_id provided)
    if supabase is not None and user_id:
        try:
            insert_payload = {
                "user_id": user_id,
                "topic": topic,
                "source_type": source_type,
                "source_name": source_name,
                "original_text": text,
                "summary": summary,
                "key_topics": key_topics,
                "key_points": key_points,
                "flashcards": flashcards,
                "quiz": quiz,
                "quiz_stats": {
                    "history": [],
                    "per_question": {},
                    "last_unsolved": [],
                },
            }
            resp = supabase.table("study_materials").insert(insert_payload).execute()
            print("✅ Inserted into study_materials:", resp.data)
            if resp.data and isinstance(resp.data, list):
                material_id = resp.data[0].get("id")
        except Exception as e:
            print("Error inserting into study_materials:", e)
    else:
        if not user_id:
            print("⚠️ No user_id provided; skipping Supabase insert.")
        if supabase is None:
            print("⚠️ Supabase client not initialized; skipping DB insert.")

    return jsonify(
        {
            "material_id": material_id,
            "text": text,
            "summary": summary,
            "key_topics": key_topics,
            "key_points": key_points,
            "flashcards": flashcards,
            "quiz": quiz,
            "notes": notes,
        }
    )


@app.post("/api/recommend_videos")
def recommend_videos():
    """
    Recommend educational YouTube videos.
    (No DB storage – just returns a list of videos.)
    """
    data = request.get_json(silent=True) or request.form or {}
    max_results = int(data.get("max_results") or 6)

    key_topics = data.get("key_topics")
    key_points = data.get("key_points")

    kp_list: List[str] = []
    if isinstance(key_topics, list) and len(key_topics) > 0:
        kp_list = [str(k).strip() for k in key_topics if str(k).strip()]
    elif isinstance(key_points, list) and len(key_points) > 0:
        kp_list = [str(k).strip() for k in key_points if str(k).strip()]

    if len(kp_list) > 8:
        kp_list = kp_list[:8]

    query_str = (data.get("query") or data.get("text") or "").strip()

    if not kp_list and query_str:
        gemini_topics = gemini_key_points_from_text(query_str, max_points=5)
        if gemini_topics:
            kp_list = gemini_topics
            print("Gemini-derived key_points:", kp_list)

    youtube_key = YOUTUBE_API_KEY

    def _variants_for_phrase(phrase: str) -> List[str]:
        p = phrase.strip()
        if not p:
            return []
        base = p.lower()
        return [
            f"{base} tutorial",
            f"{base} explained",
            f"{base} lecture",
            f"{base} crash course",
            f"{base} for beginners",
            f"{base} overview",
            f"{base} full course",
        ]

    KEEP = [
        "tutorial",
        "course",
        "lesson",
        "lecture",
        "explained",
        "learn",
        "introduction",
        "guide",
        "how to",
        "overview",
        "crash course",
    ]
    BAN = ["funny", "shorts", "reaction", "music", "meme", "song", "asmr", "review (unboxing)"]

    def _is_relevant(video_obj: Dict) -> bool:
        title = (video_obj.get("title") or "").lower()
        desc = (video_obj.get("description") or "").lower()
        text = f"{title} {desc}"
        if any(b in text for b in BAN):
            return False
        return any(k in text for k in KEEP)

    def _likecount_key(v: Dict) -> int:
        stats = v.get("statistics") or {}
        lc = stats.get("likeCount") or v.get("likeCount")
        try:
            return int(lc)
        except Exception:
            return 0

    MIN_SECONDS = 120  # 2 minutes

    if not youtube_key:
        fallback_query = query_str or (", ".join(kp_list) if kp_list else "")
        vids = curated_for_query(fallback_query, max_results=max_results * 2)
        long_enough = []
        for v in vids:
            seconds = parse_iso8601_duration(v.get("duration"))
            if seconds >= MIN_SECONDS:
                long_enough.append(v)
        if not long_enough:
            long_enough = vids
        long_enough.sort(key=_likecount_key, reverse=True)
        result = []
        for v in long_enough[:max_results]:
            v = dict(v)
            v["matched_keyword"] = None
            if v.get("videoId") and not v.get("url"):
                v["url"] = f"https://www.youtube.com/watch?v={v['videoId']}"
            result.append(v)
        return jsonify({"videos": result})

    all_videos_by_id: Dict[str, Dict] = {}

    def _filter_candidates(items: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for v in items:
            if not v.get("videoId"):
                continue
            seconds = parse_iso8601_duration(v.get("duration"))
            if seconds < MIN_SECONDS:
                continue
            if not _is_relevant(v):
                continue
            out.append(v)
        return out

    if kp_list:
        used_ids: Set[str] = set()
        keyword_best: Dict[str, Dict] = {}

        for kw in kp_list:
            if not kw.strip():
                continue

            candidates: List[Dict] = []
            for q in _variants_for_phrase(kw):
                try:
                    items = fetch_youtube_videos(q, max_results=6, api_key=youtube_key) or []
                except Exception as e:
                    print("YouTube fetch error for query:", q, str(e))
                    continue

                for v in items:
                    vid = v.get("videoId")
                    if not vid:
                        continue
                    if vid not in all_videos_by_id:
                        all_videos_by_id[vid] = v

                candidates.extend(_filter_candidates(items))

                if len(candidates) >= 12:
                    break

            if not candidates:
                continue

            uniq: Dict[str, Dict] = {}
            for v in candidates:
                vid = v.get("videoId")
                if vid and vid not in uniq:
                    uniq[vid] = v

            best: Optional[Dict] = None
            for v in uniq.values():
                vid = v.get("videoId")
                if not vid or vid in used_ids:
                    continue
                if best is None or _likecount_key(v) > _likecount_key(best):
                    best = v

            if best is not None:
                vid = best.get("videoId")
                used_ids.add(vid)
                keyword_best[kw] = best

        result: List[Dict] = []
        for kw in kp_list:
            v = keyword_best.get(kw)
            if not v:
                continue
            v = dict(v)
            v["matched_keyword"] = kw
            if v.get("videoId") and not v.get("url"):
                v["url"] = f"https://www.youtube.com/watch?v={v['videoId']}"
            result.append(v)

        if len(result) < max_results:
            leftovers: List[Dict] = []
            for vid, v in all_videos_by_id.items():
                if vid in {rv["videoId"] for rv in result if rv.get("videoId")}:
                    continue
                seconds = parse_iso8601_duration(v.get("duration"))
                if seconds < MIN_SECONDS:
                    continue
                if not _is_relevant(v):
                    continue
                leftovers.append(v)

            leftovers.sort(key=_likecount_key, reverse=True)
            for v in leftovers:
                if len(result) >= max_results:
                    break
                vv = dict(v)
                vv["matched_keyword"] = None
                if vv.get("videoId") and not vv.get("url"):
                    vv["url"] = f"https://www.youtube.com/watch?v={vv['videoId']}"
                result.append(vv)
    else:
        collected: List[Dict] = []

        search_queries: List[str] = []
        if query_str:
            search_queries = _variants_for_phrase(query_str) or [query_str]
        else:
            vids = curated_for_query("", max_results=max_results * 2)
            long_enough = [v for v in vids if parse_iso8601_duration(v.get("duration")) >= MIN_SECONDS]
            if not long_enough:
                long_enough = vids
            long_enough.sort(key=_likecount_key, reverse=True)
            result = []
            for v in long_enough[:max_results]:
                v = dict(v)
                v["matched_keyword"] = None
                if v.get("videoId") and not v.get("url"):
                    v["url"] = f"https://www.youtube.com/watch?v={v['videoId']}"
                result.append(v)
            return jsonify({"videos": result})

        for q in search_queries:
            try:
                items = fetch_youtube_videos(q, max_results=6, api_key=youtube_key) or []
            except Exception as e:
                print("YouTube fetch error for query:", q, str(e))
                continue
            collected.extend(_filter_candidates(items))
            if len(collected) >= max_results * 4:
                break

        uniq: Dict[str, Dict] = {}
        for v in collected:
            vid = v.get("videoId")
            if vid and vid not in uniq:
                uniq[vid] = v

        videos = list(uniq.values())
        if not videos:
            vids = curated_for_query(query_str, max_results=max_results * 2)
            long_enough = [v for v in vids if parse_iso8601_duration(v.get("duration")) >= MIN_SECONDS]
            if not long_enough:
                long_enough = vids
            videos = long_enough

        videos.sort(key=_likecount_key, reverse=True)
        result = []
        for v in videos[:max_results]:
            v = dict(v)
            v["matched_keyword"] = None
            if v.get("videoId") and not v.get("url"):
                v["url"] = f"https://www.youtube.com/watch?v={v['videoId']}"
            result.append(v)

    result = result[:max_results]
    print("Final video list:", result)
    return jsonify({"videos": result})


@app.post("/api/ask_question")
def ask_question():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    question = (data.get("question") or "").strip()

    if not text or not question:
        return jsonify({"error": "Missing text or question"}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500

    client = genai.Client(api_key=api_key)
    prompt = (
        "Answer the question using ONLY the PDF text below.\n\n"
        f"Question: {question}\n\n"
        f"PDF TEXT:\n{trim(text)[:40000]}"
    )

    try:
        resp = client.models.generate_content(
            model=data.get("model", "gemini-2.5-flash"),
            contents=prompt,
        )
        answer = getattr(resp, "text", "").strip() or "No answer found."
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/mindmap")
def mindmap():
    data = request.get_json(silent=True) or {}
    summary = data.get("summary") or ""
    key_topics = data.get("key_topics") or []
    key_points = data.get("key_points") or []

    result = build_mindmap(summary, key_topics, key_points)
    return jsonify(result)


@app.post("/api/mock_test")
def mock_test():
    data = request.get_json(silent=True) or {}

    raw_text = (data.get("text") or "").strip()
    topic = (data.get("topic") or "").strip() or None
    pattern_in = data.get("pattern") or []

    if not raw_text:
        summary = data.get("summary") or ""
        key_points = data.get("key_points") or []
        if isinstance(key_points, list):
            key_points_str = "\n".join(f"- {kp}" for kp in key_points)
        else:
            key_points_str = ""
        raw_text = (summary + "\n\n" + key_points_str).strip()

    if not raw_text:
        return jsonify({"error": "No text or summary provided"}), 400

    default_pattern = [
        {"marks": 10, "count": 3},
        {"marks": 5, "count": 4},
        {"marks": 3, "count": 10},
        {"marks": 2, "count": 5},
        {"marks": 1, "count": 10},
    ]

    if not isinstance(pattern_in, list) or not pattern_in:
        pattern_raw = default_pattern
    else:
        pattern_raw = pattern_in

    cleaned: List[Dict[str, int]] = []
    for p in pattern_raw:
        try:
            m = int(p.get("marks", 0))
            c = int(p.get("count", 0))
        except Exception:
            continue
        if m <= 0 or c <= 0:
            continue
        cleaned.append({"marks": m, "count": c})

    if not cleaned:
        return jsonify(
            {"error": "Invalid pattern; provide positive 'marks' and 'count'"}
        ), 400

    result = generate_mock_test(raw_text, cleaned, topic)

    if result.get("error"):
        return jsonify(result), 500

    return jsonify(result)


# -------------------------------------------------------------------
# REVISION QUIZ (select questions needing revision)
# -------------------------------------------------------------------
@app.route("/api/revision_quiz", methods=["POST"])
def revision_quiz():
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 500

    data = request.get_json() or {}
    user_id = data.get("user_id")
    material_id = data.get("material_id")
    limit = int(data.get("limit", 10))

    if not user_id or not material_id:
        return jsonify({"error": "user_id and material_id are required"}), 400

    # 1) Load study_material row (quiz + quiz_stats)
    try:
        sm_resp = (
            supabase.table("study_materials")
            .select("id, quiz, quiz_stats")
            .eq("id", material_id)
            .single()
            .execute()
        )
    except Exception as e:
        print("revision_quiz supabase exception:", e)
        return jsonify({"error": "Database error while loading material"}), 500

    sm = getattr(sm_resp, "data", sm_resp) or {}
    if not sm:
        print("revision_quiz: no study_material row for id", material_id)
        return jsonify({"error": "Study material not found"}), 404

    quiz = sm.get("quiz") or []
    quiz_stats = sm.get("quiz_stats") or {}
    per_question = quiz_stats.get("per_question", {})

    # 2) Choose which questions need revision
    revision_indices: List[int] = []
    for idx_str, stats in per_question.items():
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            continue

        attempts = stats.get("attempts", 0) or 0
        correct = stats.get("correct", 0) or 0

        if attempts == 0:
            continue  # never attempted → not a “revision” question yet

        accuracy = correct / attempts if attempts > 0 else 0.0
        needs_revision = (attempts >= 1 and correct == 0) or (
            attempts >= 3 and accuracy < 0.5
        )

        if needs_revision and 0 <= idx < len(quiz):
            revision_indices.append(idx)

    revision_indices = revision_indices[:limit]
    revision_questions = [quiz[i] for i in revision_indices]

    return jsonify(
        {
            "revision_questions": revision_questions,
            "revision_indices": revision_indices,
            "stats": quiz_stats,
        }
    )


# -------------------------------------------------------------------
# QUIZ PERFORMANCE (normal + revision submit)
# -------------------------------------------------------------------
@app.route("/api/quiz_performance", methods=["POST"])
def quiz_performance():
    """
    Store ONE quiz attempt in quiz_performance
    and update study_materials.quiz_stats (history, per_question, last_unsolved).

    Supports two modes:
      - mode="normal"  (default)  -> uses ALL quiz questions as denominator
      - mode="revision"          -> uses ONLY the given question_indices
    """
    data = request.get_json(force=True, silent=True) or {}

    user_id = data.get("user_id")
    material_id = data.get("material_id")
    answers = data.get("answers") or {}

    # NEW: mode & question_indices
    mode = (data.get("mode") or "normal").lower()
    raw_indices = data.get("question_indices") or None  # e.g. [0,2,4]

    if not user_id or not material_id:
        return jsonify({"error": "user_id and material_id are required"}), 400

    # 1) Load study material (quiz + existing stats)
    sm_resp = (
        supabase.table("study_materials")
        .select("id, quiz, quiz_stats")
        .eq("id", material_id)
        .single()
        .execute()
    )

    sm = getattr(sm_resp, "data", sm_resp)
    if not sm:
        return jsonify({"error": "Study material not found"}), 404

    quiz = sm.get("quiz") or []
    if not quiz:
        return jsonify({"error": "No quiz found for this material"}), 400

    quiz_stats = sm.get("quiz_stats") or {}
    history = quiz_stats.get("history") or []
    per_question = quiz_stats.get("per_question") or {}
    last_unsolved = quiz_stats.get("last_unsolved") or []

    # ---------- 2) Decide which questions belong to THIS attempt ----------
    n_quiz = len(quiz)

    if mode == "revision" and isinstance(raw_indices, list) and raw_indices:
        attempt_indices: list[int] = []
        for r in raw_indices:
            try:
                idx = int(r)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n_quiz:
                attempt_indices.append(idx)
        # de-duplicate while keeping order
        seen = set()
        attempt_indices = [i for i in attempt_indices if not (i in seen or seen.add(i))]
    else:
        # normal quiz → all questions
        attempt_indices = list(range(n_quiz))

    if not attempt_indices:
        return jsonify({"error": "No valid question indices for this attempt"}), 400

    # *** THIS is now the denominator for score / accuracy ***
    total_questions = len(attempt_indices)

    # ---------- 3) Compute stats for THIS attempt ----------
    attempted = 0
    skipped = 0
    correct = 0
    wrong = 0
    details = []

    for idx in attempt_indices:
        q = quiz[idx]
        idx_str = str(idx)

        selected = answers.get(idx_str)
        correct_answer = q.get("answer")

        if selected is None:
            skipped += 1
            is_correct = False
        else:
            attempted += 1
            if selected == correct_answer:
                correct += 1
                is_correct = True
            else:
                wrong += 1
                is_correct = False

        # update per_question aggregate stats (global, across normal+revision)
        q_stats = per_question.get(
            idx_str, {"correct": 0, "skipped": 0, "attempts": 0}
        )
        q_stats["attempts"] = q_stats.get("attempts", 0) + 1
        if selected is None:
            q_stats["skipped"] = q_stats.get("skipped", 0) + 1
        elif is_correct:
            q_stats["correct"] = q_stats.get("correct", 0) + 1
        per_question[idx_str] = q_stats

        details.append(
            {
                "index": idx,
                "question": q.get("question"),
                "options": q.get("options"),
                "correct_answer": correct_answer,
                "selected_answer": selected,
                "is_correct": is_correct,
            }
        )

    # ---------- 4) Score & accuracy ----------
    # score = number of correct answers in THIS attempt
    score = float(correct)

    # 🔁 IMPORTANT: denominator = number of questions in this attempt
    # (revision → number of revision questions, normal → full quiz)
    accuracy = float(correct) / float(total_questions) if total_questions > 0 else 0.0

    # ---------- 5) Append to history ----------
    attempt_no = len(history) + 1
    history_item = {
        "mode": mode,  # for debugging / dashboard breakdown if you like
        "score": score,
        "correct": correct,
        "wrong": wrong,
        "skipped": skipped,
        "attempted": attempted,
        "total_questions": total_questions,
        "created_at": datetime.utcnow().isoformat(),
        "attempt_no": attempt_no,
    }
    history.append(history_item)

    # ---------- 6) Rebuild last_unsolved from UPDATED per_question ----------
    new_last_unsolved = []
    for idx, q in enumerate(quiz):
        idx_str = str(idx)
        stats_for_q = per_question.get(idx_str, {})
        attempts_q = stats_for_q.get("attempts", 0)
        correct_q = stats_for_q.get("correct", 0)

        # "unsolved" = tried at least once and never got correct
        if attempts_q > 0 and correct_q == 0:
            new_last_unsolved.append(
                {
                    "index": idx,
                    "question": q.get("question"),
                    "options": q.get("options"),
                    "correct_answer": q.get("answer"),
                    "selected_answer": None,
                }
            )

    quiz_stats = {
        "history": history,
        "per_question": per_question,
        "last_unsolved": new_last_unsolved,
    }

    # ---------- 7) UPDATE study_materials.quiz_stats ----------
    supabase.table("study_materials") \
        .update({"quiz_stats": quiz_stats}) \
        .eq("id", material_id) \
        .execute()

    # ---------- 8) INSERT into quiz_performance ----------
    qp_row = {
        "user_id": user_id,
        "material_id": material_id,
        "correct_answers": correct,
        "wrong_answers": wrong,
        "skipped": skipped,
        "total_questions": total_questions,  # 👈 uses attempt_indices length
        "score": score,
        "accuracy": accuracy,
        "attempt_no": attempt_no,
        "answers": answers,
        "details": details,
    }

    qp_resp = (
        supabase.table("quiz_performance")
        .insert(qp_row)
        .execute()
    )
    qp_data = getattr(qp_resp, "data", qp_resp)

    return jsonify(
        {
            "ok": True,
            "attempt": history_item,
            "quiz_stats": quiz_stats,
            "quiz_performance_row": qp_data,
        }
    )



# -------------------------------------------------------------------
# DASHBOARD ENDPOINTS
# -------------------------------------------------------------------
@app.route("/api/dashboard/overview/<user_id>", methods=["GET"])
def dashboard_overview(user_id):
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        qp_resp = (
            supabase.table("quiz_performance")
            .select("material_id, correct_answers, wrong_answers, created_at")
            .eq("user_id", user_id)
            .execute()
        )

        rows = qp_resp.data or []
        if not rows:
            return jsonify({"materials": [], "global_stats": {}})

        materials_summary = {}

        total_correct = 0
        total_attempted = 0

        for row in rows:
            m_id = row["material_id"]

            correct = int(row.get("correct_answers") or 0)
            wrong = int(row.get("wrong_answers") or 0)
            attempted = correct + wrong

            total_correct += correct
            total_attempted += attempted

            if m_id not in materials_summary:
                materials_summary[m_id] = {
                    "material_id": m_id,
                    "total_attempts": 0,
                    "correct": 0,
                    "attempted": 0,
                    "last_attempt_at": row.get("created_at"),
                }

            m = materials_summary[m_id]
            m["total_attempts"] += 1
            m["correct"] += correct
            m["attempted"] += attempted

            if row.get("created_at") and row["created_at"] > m["last_attempt_at"]:
                m["last_attempt_at"] = row["created_at"]

        # ---------- per material accuracy ----------
        materials = []
        for m in materials_summary.values():
            acc = (m["correct"] / m["attempted"] * 100) if m["attempted"] > 0 else 0
            materials.append({
                "material_id": m["material_id"],
                "total_attempts": m["total_attempts"],
                "avg_accuracy": round(acc, 2),
                "last_attempt_at": m["last_attempt_at"],
            })

        # ---------- GLOBAL accuracy ----------
        global_accuracy = (
            (total_correct / total_attempted) * 100
            if total_attempted > 0
            else 0
        )

        global_stats = {
            "total_materials": len(materials_summary),
            "total_attempts": len(rows),
            "avg_accuracy": round(global_accuracy, 2),
        }

        return jsonify({
            "materials": materials,
            "global_stats": global_stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/dashboard/material/<material_id>/<user_id>", methods=["GET"])
def dashboard_material(material_id, user_id):
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        # quiz performance history for this material
        qp_resp = (
            supabase.table("quiz_performance")
            .select("*")
            .eq("material_id", material_id)
            .eq("user_id", user_id)
            .order("attempt_no", desc=False)
            .execute()
        )
        qp_rows = getattr(qp_resp, "data", qp_resp) or []

        # study_materials info + quiz_stats
        sm_resp = (
            supabase.table("study_materials")
            .select("quiz_stats, topic, source_name")
            .eq("id", material_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        sm = getattr(sm_resp, "data", sm_resp) or {}

        stats = (sm.get("quiz_stats") or {}).get("per_question", {})
        last_unsolved = (sm.get("quiz_stats") or {}).get("last_unsolved", [])

        history: List[Dict[str, Any]] = []
        for row in qp_rows:
            acc = float(row.get("accuracy", 0))
            history.append(
                {
                    "attempt_no": row.get("attempt_no"),
                    "created_at": row.get("created_at"),
                    "accuracy": round(acc * 100, 2),
                    "correct": row.get("correct_answers", 0),
                    "wrong": row.get("wrong_answers", 0),
                    "skipped": row.get("skipped", 0),
                    "total_questions": row.get("correct_answers", 0) + row.get("wrong_answers", 0),
                    "mode": row.get("mode", "quiz"),
                }
            )
            print("History entry:", history[-1])
        return jsonify(
            {
                "material_info": {
                    "material_id": material_id,
                    "topic": sm.get("topic"),
                    "source_name": sm.get("source_name"),
                },
                "quiz_history": history,
                "per_question": stats,
                "last_unsolved": last_unsolved,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from collections import defaultdict

# 🔹 Overall user dashboard – all materials, all attempts
@app.get("/api/dashboard/user/<user_id>")
def user_dashboard(user_id):
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        qp = (
    supabase.table("quiz_performance")
    .select(
        """
        material_id,
        created_at,
        attempt_no,
        correct_answers,
        wrong_answers,
        skipped,
        study_materials!quiz_performance_material_id_fkey (
            source_name
        )
        """
    )
    .eq("user_id", user_id)
    .order("created_at", desc=False)
    .execute()
)

         
    except Exception as e:
        print("user_dashboard supabase error:", e)
        return jsonify({"error": "Database error"}), 500
    print("Quiz performance response:", qp)
    rows = getattr(qp, "data", []) or []
    if not rows:
        # empty skeleton response
        empty_summary = {
            "total_attempts": 0,
            "distinct_materials": 0,
            "overall_accuracy": 0.0,
            "avg_score": 0.0,
            "avg_accuracy": 0.0,
            "best_score": 0.0,
            "best_accuracy": 0.0,
        }
        return jsonify({"summary": empty_summary, "attempts": [], "by_date": []})

    # ---------- aggregation variables ----------
    total_attempts = len(rows)
    material_ids = set()

    overall_correct = 0          # for OVERALL accuracy
    overall_total_q = 0

    sum_scores = 0.0             # for avg score
    sum_accuracy_frac = 0.0      # sum of per-attempt accuracy (0–1 values)

    best_score = 0.0
    best_accuracy_frac = 0.0

    attempts_out = []
    by_date_map = {}             # date => {attempts, sum_accuracy_frac, sum_score}

    # ---------- loop over attempts ----------
    for row in rows:
        mid = row["material_id"]
        material_ids.add(mid)
        material_name = None
        sm = row.get("study_materials")
        if isinstance(sm, dict):
          material_name = sm.get("source_name")[:15]
        correct = int(row.get("correct_answers") or 0)
        wrong = int(row.get("wrong_answers") or 0)
        skipped = int(row.get("skipped") or 0)

        total_q = correct + wrong 
        score = float(correct)  # 1 point per correct (same as quiz_performance)

        # 🧮 accuracy for THIS attempt (ALWAYS re-computed)
        if total_q > 0:
            accuracy_frac = correct / total_q  # e.g. 0.6, 0.5, 1.0
        else:
            accuracy_frac = 0.0

        # accumulate for overall + averages
        overall_correct += correct
        overall_total_q += total_q
        sum_scores += score
        sum_accuracy_frac += accuracy_frac

        if score > best_score:
            best_score = score
        if accuracy_frac > best_accuracy_frac:
            best_accuracy_frac = accuracy_frac

        created_at = row.get("created_at") or ""
        date_key = created_at[:10]  # "YYYY-MM-DD"

        d = by_date_map.setdefault(
            date_key,
            {"attempts": 0, "sum_accuracy_frac": 0.0, "sum_score": 0.0},
        )
        d["attempts"] += 1
        d["sum_accuracy_frac"] += accuracy_frac
        d["sum_score"] += score
        print("Row material:", row.get("study_materials"))

        attempts_out.append(
            {
                "attempt_no": int(row.get("attempt_no") or 0),
                "material_id": mid,
                "material_name": material_name or "Untitled Material",
                "created_at": created_at,
                "score": score,
                "total_questions": total_q,
                "accuracy": accuracy_frac * 100.0,  # send as percentage
                "correct": correct,
                "wrong": wrong,
                "skipped": skipped,
            }
        )

    # ---------- final summary numbers ----------
    if overall_total_q > 0:
        overall_accuracy = (overall_correct / overall_total_q) * 100.0
    else:
        overall_accuracy = 0.0

    avg_score = sum_scores / total_attempts if total_attempts > 0 else 0.0
    avg_accuracy = (
        (sum_accuracy_frac / total_attempts) * 100.0 if total_attempts > 0 else 0.0
    )

    # build per-date rows
    by_date_out = []
    for date_key, d in sorted(by_date_map.items()):
        attempts_d = d["attempts"]
        if attempts_d > 0:
            avg_acc_d = (d["sum_accuracy_frac"] / attempts_d) * 100.0
            avg_score_d = d["sum_score"] / attempts_d
        else:
            avg_acc_d = 0.0
            avg_score_d = 0.0

        by_date_out.append(
            {
                "date": date_key,
                "attempts": attempts_d,
                "avg_score": avg_score_d,
                "avg_accuracy": avg_acc_d,
            }
        )

    summary = {
        "total_attempts": total_attempts,
        "distinct_materials": len(material_ids),
        "overall_accuracy": round(overall_accuracy, 1),
        "avg_score": round(avg_score, 2),
        "avg_accuracy": round(avg_accuracy, 2),
        "best_score": round(best_score, 2),
        "best_accuracy": round(best_accuracy_frac * 100.0, 1),
    }
    print("User dashboard summary:", summary)
    print("Total attempts:", total_attempts)
    return jsonify(
        {
            "summary": summary,
            "attempts": attempts_out,
            "by_date": by_date_out,
        }
    )



# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "0").lower() in ("1", "true", "on")

    print("\n==== FLASK ROUTES ====")
    for rule in app.url_map.iter_rules():
        print(rule.endpoint, rule.rule, list(rule.methods))
    print("=======================\n")
    print(f"Backend running → http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
