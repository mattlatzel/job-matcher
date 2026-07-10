"""
CV Job Matcher — Backend
Run: python main.py
Requires: ANTHROPIC_API_KEY and JSEARCH_API_KEY set as environment variables
"""

import asyncio
import io
import json
import os
import re
import uuid
from pathlib import Path

import anthropic
import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="CV Job Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  { session_id: { status, profile, results, total_jobs, error } }
sessions: dict = {}

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")

SONNET = "claude-sonnet-5"   # profile extraction — needs quality reasoning
HAIKU  = "claude-haiku-4-5-20251001"   # job scoring — runs ~20+ times, needs speed


# ── CV Parsing ───────────────────────────────────────────────────────────────

def parse_pdf(content: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_docx(content: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


# ── Claude: Quick title extraction (Haiku, ~0.5s) ────────────────────────────

async def quick_extract_title(client: anthropic.AsyncAnthropic, cv_text: str) -> str:
    resp = await client.messages.create(
        model=HAIKU,
        max_tokens=30,
        messages=[{"role": "user", "content":
            f"Extract only the job title from this CV. Return the title only, nothing else.\n\n{cv_text[:1500]}"}]
    )
    return next(b.text for b in resp.content if hasattr(b, "text")).strip()[:80]


# ── Claude: Extract structured profile from CV ───────────────────────────────

async def extract_profile(cv_text: str) -> dict:
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=SONNET,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Analyze this CV and extract a structured profile. Return JSON only, no prose.

CV TEXT:
{cv_text[:8000]}

Return exactly this JSON structure:
{{
  "current_title": "most recent or target job title",
  "seniority": "junior | mid | senior | lead | executive",
  "years_experience": <number>,
  "core_domain": "1-sentence description of their main professional domain",
  "core_skills": ["top 6 most important skills"],
  "job_search_query": "best search query to find matching jobs (role + domain, e.g. 'Senior Product Manager SaaS')",
  "adjacent_titles": ["2-3 alternative job titles this person could also qualify for, e.g. Head of Product, Director of Product"],
  "location": "city/country or Remote",
  "summary": "2-sentence professional summary"
}}"""
        }]
    )
    text = next(b.text for b in resp.content if hasattr(b, "text")).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group() if match else text)


# ── JSearch: Fetch jobs ───────────────────────────────────────────────────────

async def _fetch_query(client: httpx.AsyncClient, query: str, pages: int = 3) -> list[dict]:
    """Fetch up to `pages` pages for a single query string."""
    jobs = []
    for page in range(1, pages + 1):
        try:
            resp = await client.get(
                "https://jsearch.p.rapidapi.com/search-v2",
                params={
                    "query": query,
                    "page": str(page),
                    "num_pages": "1",
                    "date_posted": "month",
                    "country": "gb",
                },
                headers={
                    "X-RapidAPI-Key": JSEARCH_API_KEY,
                    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
                },
            )
            data = resp.json()
            raw = data.get("data", [])
            batch = raw.get("jobs", []) if isinstance(raw, dict) else raw
            jobs.extend(batch)
            if not batch:
                break
        except Exception as e:
            print(f"  JSearch error ({query}, page {page}): {e}")
            break
    return jobs


async def _fetch_query_standalone(query: str, pages: int = 2) -> list[dict]:
    """Wrapper that creates its own httpx client — for parallel calls."""
    async with httpx.AsyncClient(timeout=30) as client:
        return await _fetch_query(client, query, pages)


async def fetch_jobs(profile: dict) -> list[dict]:
    title           = profile.get("current_title", "professional")
    domain          = profile.get("job_search_query") or title
    adjacent_titles = profile.get("adjacent_titles", [])[:2]  # max 2 adjacent

    # Build all queries
    queries = [
        (f"{domain} in London",  4),   # specific + domain, most pages
        (f"{title} in London",   2),   # title only, fewer pages
    ]
    for adj in adjacent_titles:
        queries.append((f"{adj} in London", 2))

    print(f"  Queries: {[q for q, _ in queries]}")

    async with httpx.AsyncClient(timeout=45) as client:
        results = await asyncio.gather(*[
            _fetch_query(client, q, pages=p) for q, p in queries
        ])

    # Merge and deduplicate
    seen: set = set()
    all_jobs: list[dict] = []
    for batch in results:
        for job in batch:
            jid = job.get("job_id")
            if jid and jid not in seen:
                seen.add(jid)
                all_jobs.append(job)

    print(f"  Total unique jobs fetched: {len(all_jobs)}")
    return all_jobs


# ── Claude: Pre-filter jobs by title only ────────────────────────────────────

async def prefilter_jobs(
    client: anthropic.AsyncAnthropic,
    profile: dict,
    jobs: list[dict],
) -> list[dict]:
    """
    Quickly eliminate obvious mismatches using only job titles + companies.
    Returns only the jobs worth scoring in detail.
    """
    batch_size = 20
    passed: list[dict] = []

    profile_line = (
        f"{profile.get('seniority', '').capitalize()} {profile.get('current_title')} "
        f"with {profile.get('years_experience')} years in {profile.get('core_domain')}"
    )

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        titles_block = "\n".join(
            f"{j+1}. {job.get('job_title', 'N/A')} — {job.get('employer_name', 'N/A')}"
            for j, job in enumerate(batch)
        )

        resp = await client.messages.create(
            model=HAIKU,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""You are a fast job screener. Decide which jobs are worth evaluating for this candidate.

CANDIDATE: {profile_line}

JOBS:
{titles_block}

Return a JSON array of "yes" or "no" for each job (in order).
"yes" = plausibly relevant based on title/seniority alone.
"no"  = clearly wrong level, wrong field, or irrelevant.

JSON array only. Example: ["yes","no","yes"]"""
            }]
        )

        text = next(b.text for b in resp.content if hasattr(b, "text")).strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        decisions: list[str] = json.loads(match.group() if match else text)

        for j, job in enumerate(batch):
            if j < len(decisions) and decisions[j].lower() == "yes":
                passed.append(job)

    return passed


# ── Claude: Score a batch of jobs against the profile ────────────────────────

async def score_batch(
    client: anthropic.AsyncAnthropic,
    profile: dict,
    jobs: list[dict],
) -> list[dict]:
    jobs_block = ""
    for i, job in enumerate(jobs, 1):
        desc = (job.get("job_description") or "")[:3000]
        jobs_block += f"""
JOB {i}
Title: {job.get('job_title', 'N/A')}
Company: {job.get('employer_name', 'N/A')}
Location: {job.get('job_city', '')} {job.get('job_country', '')} {'(Remote)' if job.get('job_is_remote') else ''}
Description:
{desc}
────────────────────
"""

    profile_block = f"""Current title: {profile.get('current_title')}
Seniority: {profile.get('seniority')}
Years of experience: {profile.get('years_experience')}
Core domain: {profile.get('core_domain')}
Core skills: {', '.join(profile.get('core_skills', []))}
Summary: {profile.get('summary')}"""

    resp = await client.messages.create(
        model=SONNET,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""You are a senior recruiter evaluating job fit. Score each job and provide a structured breakdown.

Scoring rules:
- Reason about core experience vs core requirements — NOT keyword overlap.
- 90+: natural fit, strong alignment on seniority, domain, and skills.
- 70-89: good alignment but some gaps.
- Below 70: poor fit.
- Be honest. Do not inflate scores.

CANDIDATE:
{profile_block}

JOBS:
{jobs_block}

Return a JSON array (one object per job, in order):
[
  {{
    "score": <0-100>,
    "seniority_match": true | false,
    "domain_match": true | false,
    "strengths": ["1-3 specific strengths from the CV that match this job"],
    "gaps": ["1-3 specific gaps or missing requirements, or empty array if none"],
    "reason": "<2 sentences: overall fit summary>"
  }},
  ...
]

JSON only. No prose."""
        }]
    )

    text = next(b.text for b in resp.content if hasattr(b, "text")).strip()
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        scores: list[dict] = json.loads(match.group() if match else text)
    except Exception as e:
        print(f"  Score batch parse error: {e}")
        scores = []

    results = []
    for i, job in enumerate(jobs):
        s = scores[i] if i < len(scores) else {
            "score": 0, "seniority_match": False, "domain_match": False,
            "strengths": [], "gaps": [], "reason": "Could not evaluate."
        }

        # Build salary string if available
        salary = None
        min_s = job.get("job_min_salary")
        max_s = job.get("job_max_salary")
        currency = job.get("job_salary_currency") or "£"
        period = (job.get("job_salary_period") or "").lower()
        if min_s and max_s:
            salary = f"{currency}{int(min_s):,} – {currency}{int(max_s):,}"
            if period in ("year", "annual"):
                salary += " / yr"
            elif period == "month":
                salary += " / mo"
        elif min_s:
            salary = f"From {currency}{int(min_s):,}"
        elif max_s:
            salary = f"Up to {currency}{int(max_s):,}"

        results.append({
            "title":           job.get("job_title", "Unknown Role"),
            "company":         job.get("employer_name", "Unknown Company"),
            "location":        f"{job.get('job_city', '')} {job.get('job_country', '')}".strip() or "London",
            "is_remote":       bool(job.get("job_is_remote")),
            "apply_link":      job.get("job_apply_link") or job.get("job_google_link") or "#",
            "score":           int(s.get("score", 0)),
            "seniority_match": bool(s.get("seniority_match", False)),
            "domain_match":    bool(s.get("domain_match", False)),
            "strengths":       s.get("strengths", []),
            "gaps":            s.get("gaps", []),
            "reason":          s.get("reason", ""),
            "posted":          (job.get("job_posted_at_datetime_utc") or "")[:10],
            "salary":          salary,
        })
    return results


# ── Claude: Haiku quick-score (scores only, no breakdown) ────────────────────

async def quick_score_all(
    client: anthropic.AsyncAnthropic,
    profile: dict,
    jobs: list[dict],
) -> list[tuple]:
    """
    Fast Haiku pass returning (job, score) pairs sorted by score desc.
    Uses only title + company + 200-char description snippet.
    """
    profile_line = (
        f"{profile.get('current_title')}, {profile.get('years_experience')} yrs exp, "
        f"{profile.get('core_domain')}"
    )
    batch_size = 10
    scored: list[tuple] = []
    sem = asyncio.Semaphore(8)

    async def score_quick_batch(batch):
        jobs_text = "\n".join(
            f"{j+1}. {job.get('job_title','?')} @ {job.get('employer_name','?')}: "
            f"{(job.get('job_description') or '')[:200]}"
            for j, job in enumerate(batch)
        )
        async with sem:
            try:
                resp = await client.messages.create(
                    model=HAIKU,
                    max_tokens=80,
                    messages=[{"role": "user", "content":
                        f"Candidate: {profile_line}\n\nRate each job 0-100 for fit. "
                        f"JSON array of integers only.\n\n{jobs_text}\n\nScores:"}]
                )
                text = next(b.text for b in resp.content if hasattr(b, "text")).strip()
                match = re.search(r"\[.*?\]", text, re.DOTALL)
                scores = json.loads(match.group() if match else "[]")
            except Exception as e:
                print(f"  Quick score error: {e}")
                scores = []
            for j, job in enumerate(batch):
                scored.append((job, int(scores[j]) if j < len(scores) else 50))

    batches = [jobs[i:i+batch_size] for i in range(0, len(jobs), batch_size)]
    await asyncio.gather(*[score_quick_batch(b) for b in batches])
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Claude: Sonnet deep-score with streaming to session ──────────────────────

async def score_and_stream(
    session: dict,
    client: anthropic.AsyncAnthropic,
    profile: dict,
    jobs: list[dict],
) -> None:
    """Score jobs with Sonnet, appending results to session as each batch finishes."""
    session["scoring_total"] = len(jobs)
    session["scoring_done"]  = 0
    batches = [jobs[i:i+3] for i in range(0, len(jobs), 3)]
    sem = asyncio.Semaphore(4)

    async def do_batch(batch):
        async with sem:
            results = await score_batch(client, profile, batch)
            session["results"].extend(results)
            session["results"].sort(key=lambda x: x["score"], reverse=True)
            session["scoring_done"] += len(results)

    await asyncio.gather(*[do_batch(b) for b in batches])


# ── Claude: CV gap analysis across all scored jobs ────────────────────────────

async def analyze_cv_gaps(
    client: anthropic.AsyncAnthropic,
    profile: dict,
    scored_jobs: list[dict],
) -> list[dict]:
    """
    Meta-analysis across top scored jobs to identify recurring CV gaps.
    Returns a list of {gap, suggestion, frequency} objects.
    """
    top = scored_jobs[:20]  # look at top 20 matched jobs

    jobs_gaps = ""
    for job in top:
        gaps = job.get("gaps", [])
        if gaps:
            jobs_gaps += f"- {job['title']} ({job['score']}% match): {'; '.join(gaps)}\n"

    if not jobs_gaps:
        return []

    profile_block = f"""Title: {profile.get('current_title')}
Seniority: {profile.get('seniority')}
Years of experience: {profile.get('years_experience')}
Core domain: {profile.get('core_domain')}
Core skills: {', '.join(profile.get('core_skills', []))}
Summary: {profile.get('summary')}"""

    resp = await client.messages.create(
        model=SONNET,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""You are a career coach analysing a candidate's CV against London job market requirements.

CANDIDATE PROFILE:
{profile_block}

GAPS IDENTIFIED ACROSS MATCHED JOBS:
{jobs_gaps}

Based on these recurring gaps, identify the 3-5 most impactful improvements this candidate should make to their CV to increase their match rate.

Be specific and actionable. Focus on gaps that appear multiple times — these are the most valuable to address.

Return the 3 most important gaps only. Be very brief — a few words each.

Return JSON array only:
[
  {{
    "gap": "3-5 word gap title",
    "action": "what to add or fix in CV (max 10 words)"
  }},
  ...
]"""
        }]
    )

    text = next(b.text for b in resp.content if hasattr(b, "text")).strip()
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        return json.loads(match.group() if match else text)
    except Exception as e:
        print(f"  Gap analysis parse error: {e}\n  Raw: {text[:200]}")
        return []


# ── Background worker ─────────────────────────────────────────────────────────

async def process_cv(session_id: str, cv_text: str) -> None:
    session = sessions[session_id]
    try:
        ai_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        # 1. Extract profile
        print("▶ Step 1: Extracting profile…")
        session["status"] = "extracting_profile"
        profile = await extract_profile(cv_text)
        session["profile"] = profile
        print(f"✓ Profile: {profile.get('current_title')}")

        # 2. Fetch jobs
        print("▶ Step 2: Fetching jobs…")
        session["status"] = "fetching_jobs"
        jobs = await fetch_jobs(profile)
        session["total_jobs"] = len(jobs)
        print(f"✓ Fetched {len(jobs)} jobs")

        # 3. Pre-filter by title
        print("▶ Step 3: Pre-filtering…")
        session["status"] = "scoring"
        jobs = await prefilter_jobs(ai_client, profile, jobs)
        print(f"✓ Pre-filter passed: {len(jobs)} jobs")

        # 4. Sonnet deep scoring
        print("▶ Step 4: Deep scoring…")
        batches = [jobs[i:i+3] for i in range(0, len(jobs), 3)]
        sem = asyncio.Semaphore(6)

        async def score_with_sem(batch):
            async with sem:
                return await score_batch(ai_client, profile, batch)

        batch_results = await asyncio.gather(*[score_with_sem(b) for b in batches])
        all_results = [r for br in batch_results for r in br]
        all_results.sort(key=lambda x: x["score"], reverse=True)
        session["results"] = all_results
        print(f"✓ Scored {len(all_results)} jobs")

        # 5. Gap analysis
        print("▶ Step 5: CV gap analysis…")
        session["gap_analysis"] = await analyze_cv_gaps(ai_client, profile, all_results)
        print(f"✓ Found {len(session['gap_analysis'])} gaps")

        session["status"] = "done"
        print(f"✓ Done! {len([r for r in all_results if r['score'] >= 70])} jobs ≥70%")

    except Exception as exc:
        print(f"✗ ERROR: {exc}")
        import traceback; traceback.print_exc()
        session["status"] = "error"
        session["error"] = str(exc)


# ── API Routes ────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    if not JSEARCH_API_KEY:
        raise HTTPException(500, "JSEARCH_API_KEY not set")

    content = await file.read()
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        cv_text = parse_pdf(content)
    elif name.endswith(".docx"):
        cv_text = parse_docx(content)
    elif name.endswith(".txt"):
        cv_text = content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Unsupported file type. Please upload PDF, DOCX, or TXT.")

    if len(cv_text.strip()) < 100:
        raise HTTPException(400, "Could not extract text from the CV. Try a different format.")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "status":       "starting",
        "profile":      None,
        "results":      [],
        "gap_analysis": [],
        "total_jobs":   0,
        "error":        None,
        "docx_bytes":   content if name.endswith(".docx") else None,
    }
    background_tasks.add_task(process_cv, session_id, cv_text)
    return {"session_id": session_id, "is_docx": name.endswith(".docx")}


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[session_id]
    return {
        "status":        s["status"],
        "total_jobs":    s["total_jobs"],
        "profile":       s["profile"],
        "error":         s["error"],
    }


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[session_id]
    return {
        "jobs":         s.get("results", []),
        "gap_analysis": s.get("gap_analysis", []),
    }


# ── Claude: Improve CV based on gap analysis ─────────────────────────────────

async def improve_cv_content(
    client: anthropic.AsyncAnthropic,
    docx_bytes: bytes,
    gap_analysis: list[dict],
    profile: dict,
) -> bytes:
    import docx as docx_lib

    doc = docx_lib.Document(io.BytesIO(docx_bytes))

    # Extract non-empty paragraphs with their index
    para_lines = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            para_lines.append(f"{i}: {text}")

    paragraphs_block = "\n".join(para_lines)

    gaps_block = "\n".join(
        f"- Gap: {g.get('gap')} | Action: {g.get('action')}" for g in gap_analysis
    )

    resp = await client.messages.create(
        model=SONNET,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""You are a professional CV editor. Your task is to improve specific lines in this CV to address identified skill gaps.

CANDIDATE PROFILE:
Title: {profile.get('current_title')}
Domain: {profile.get('core_domain')}
Skills: {', '.join(profile.get('core_skills', []))}

GAPS TO ADDRESS:
{gaps_block}

CV PARAGRAPHS (format: index: text):
{paragraphs_block}

Instructions:
- Identify which paragraphs should be rewritten to address the gaps above.
- Only change paragraphs where an improvement is genuinely needed — do NOT change headings, contact info, dates, or sections unrelated to the gaps.
- Rewrite the selected paragraphs to naturally incorporate the missing skills or experience framing.
- Keep the same general length and tone. Do not invent experience that isn't there — reframe and strengthen what exists.
- Return ONLY the paragraphs you changed.

Return JSON array only:
[
  {{"index": <paragraph_index>, "new_text": "improved paragraph text"}},
  ...
]"""
        }]
    )

    text = next(b.text for b in resp.content if hasattr(b, "text")).strip()
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        changes: list[dict] = json.loads(match.group() if match else text)
    except Exception as e:
        print(f"  CV improve parse error: {e}")
        changes = []

    print(f"  Applying {len(changes)} paragraph changes to CV")

    for change in changes:
        idx = change.get("index")
        new_text = change.get("new_text", "").strip()
        if idx is None or not new_text:
            continue
        if idx >= len(doc.paragraphs):
            continue
        para = doc.paragraphs[idx]
        if not para.runs:
            # No runs — just set text directly (rare)
            para.clear()
            para.add_run(new_text)
        else:
            # Preserve first run's character formatting, wipe the rest
            para.runs[0].text = new_text
            for run in para.runs[1:]:
                run.text = ""

    out = io.BytesIO()
    doc.save(out)
    out.seek(0)
    return out.read()


@app.post("/api/improve-cv/{session_id}")
async def improve_cv(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[session_id]
    if s.get("status") != "done":
        raise HTTPException(400, "Analysis not complete yet")
    if not s.get("docx_bytes"):
        raise HTTPException(400, "No Word document in this session")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    improved_bytes = await improve_cv_content(
        client,
        s["docx_bytes"],
        s.get("gap_analysis", []),
        s.get("profile", {}),
    )

    return StreamingResponse(
        io.BytesIO(improved_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=improved_cv.docx"},
    )


@app.get("/")
async def serve_frontend():
    html = (Path(__file__).parent / "index.html").read_text()
    return HTMLResponse(content=html)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀  CV Job Matcher running at http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
