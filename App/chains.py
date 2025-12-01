import os
import json
import re
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from utils import truncate_text_by_chars
try:
    from utils import select_top_resume_sections
except Exception:
    select_top_resume_sections = None

load_dotenv()


def _looks_like_name_line(line: str) -> bool:
    """
    Heuristic: detect short lines likely to be a person's name.
    - 1-3 words
    - contains capital letters
    - reasonable length (< 60 chars)
    """
    if not line:
        return False
    line = line.strip()
    if len(line) > 60:
        return False
    words = line.split()
    if not (1 <= len(words) <= 4):
        return False
    cap_words = sum(1 for w in words if any(ch.isupper() for ch in w))
    return cap_words >= max(1, len(words) // 2)


class Chain:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found in environment. Set it in a .env file or environment variables."
            )

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
        )

    # --------- helper: extract candidate name from resume -----------
    def _extract_name_from_resume(self, resume_text: str) -> str:
        """
        Best-effort extraction of candidate name from resume text.
        Strategy:
          - Look at the first non-empty lines (first 6) and pick the one that looks like a name.
          - If nothing found, try to find lines that appear like 'Name: John Doe' and extract.
          - Return empty string if not found.
        """
        if not resume_text:
            return ""

        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
        # look at first few lines for a name-like line
        for i in range(min(len(lines), 6)):
            if _looks_like_name_line(lines[i]):
                return lines[i]

        # look for 'Name: John Doe' patterns
        for ln in lines[:50]:
            m = re.search(r'\bname[:\-\s]+([A-Z][A-Za-z\'`.\- ]{1,60})', ln, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if _looks_like_name_line(candidate):
                    return candidate

        return ""

    # ---------------------------------------------------------
    # SIGNATURE ENSURER (uses the derived name)
    # ---------------------------------------------------------
    def _ensure_signature(self, email_body: str, candidate_name: str) -> str:
        """
        Make sure the final email ends with a single signature using candidate_name (if present).
        Rules:
          - If candidate_name present in the model output, don't append.
          - If model output contains a signoff phrase followed by another name, replace that name with candidate_name.
          - If no signoff present and candidate_name available, append signature.
          - If no candidate_name, return the model output (tidied).
        """
        body = (email_body or "").strip()
        if not body:
            if candidate_name:
                return f"Best regards,\n{candidate_name}"
            return ""

        low = body.lower()
        if candidate_name and candidate_name.lower() in low:
            # already present, tidy whitespace and return
            return re.sub(r'\n{3,}', '\n\n', body).strip()

        # common signoffs
        signoff_phrases = ["best regards", "regards", "sincerely", "thank you", "thanks"]
        # find last occurrence of any signoff phrase
        last_idx = -1
        found_phrase = None
        for p in signoff_phrases:
            idx = low.rfind(p)
            if idx > last_idx:
                last_idx = idx
                found_phrase = p if idx != -1 else found_phrase

        if last_idx != -1 and found_phrase:
            # split tail from phrase to end and attempt to detect/replace name-like line
            tail = body[last_idx:]
            tail_lines = tail.splitlines()
            # check next few lines for a name-like line
            for i in range(1, min(4, len(tail_lines))):
                cand = tail_lines[i].strip()
                if _looks_like_name_line(cand):
                    # replace the candidate name with our candidate_name (if provided)
                    if candidate_name:
                        pre = body[:last_idx]
                        # keep the phrase line, then append candidate_name + tidy
                        new_tail = tail_lines[0].strip() + "\n" + candidate_name
                        return re.sub(r'\n{3,}', '\n\n', (pre.strip() + "\n" + new_tail).strip())
                    else:
                        return re.sub(r'\n{3,}', '\n\n', body).strip()
            # phrase found but no name after it
            if candidate_name:
                return re.sub(r'\n{3,}', '\n\n', body.rstrip() + f"\n\n{candidate_name}").strip()
            else:
                return re.sub(r'\n{3,}', '\n\n', body).strip()

        # no signoff found
        if candidate_name:
            return re.sub(r'\n{3,}', '\n\n', body.rstrip() + f"\n\nBest regards,\n{candidate_name}").strip()

        return re.sub(r'\n{3,}', '\n\n', body).strip()

    # ---------------------------------------------------------
    # JOB EXTRACTION
    # ---------------------------------------------------------
    def extract_jobs(self, cleaned_text: str):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE OR PROVIDED JD:
            {page_data}

            ### INSTRUCTION:
            Extract job postings from the text. Return a JSON array of objects with keys:
            - role (string)
            - experience (string or range if present)
            - skills (list of strings)
            - description (string, the job description / responsibilities)

            Only return valid JSON (no explanation).
            ### JSON:
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": truncate_text_by_chars(cleaned_text, max_chars=30000)})

        try:
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
        except OutputParserException:
            try:
                parsed = json.loads(res.content)
            except Exception:
                raise OutputParserException("Failed to parse LLM output to JSON.")

        return parsed if isinstance(parsed, list) else [parsed]

    # ---------------------------------------------------------
    # PREVIEW MATCHED RESUME POINTS (optional)
    # ---------------------------------------------------------
    def preview_matched_points(self, job, resume_text: str):
        if not select_top_resume_sections:
            return []
        try:
            jd_text = job if isinstance(job, str) else (job.get("description") or job.get("role") or str(job))
            return select_top_resume_sections(resume_text, jd_text, top_k=5)
        except Exception:
            return []

    # ---------------------------------------------------------
    # EMAIL GENERATION (auto name extraction)
    # ---------------------------------------------------------
    def write_personalized_mail(self, job, resume_text: str) -> str:
        """
        Generate a personalized email for the job using resume_text.
        Candidate name is auto-extracted from resume_text (if possible).
        """
        job_description = job if isinstance(job, str) else (job.get("description") or job.get("role") or str(job))

        # try to extract candidate name from resume
        candidate_name = self._extract_name_from_resume(resume_text)

        # Select top resume points if helper exists
        selected_text = ""
        if select_top_resume_sections:
            try:
                top = select_top_resume_sections(resume_text, job_description, top_k=6)
                if top:
                    bullets = []
                    for sec, sc in top:
                        short = sec.strip()
                        if len(short) > 900:
                            short = short[:900].rstrip() + "..."
                        bullets.append(f"- {short}")
                    selected_text = "\n".join(bullets)
            except Exception:
                selected_text = ""

        if not selected_text:
            selected_text = truncate_text_by_chars(resume_text, max_chars=2000)

        # build prompt; include extracted name in instruction only if found
        if candidate_name:
            identity = f"You are {candidate_name}, a motivated and skilled candidate applying for the job above."
        else:
            identity = "You are a motivated and skilled candidate applying for the job above."

        prompt_template = f"""
        ### JOB DESCRIPTION:
        {{job_description}}

        ### CANDIDATE MOST RELEVANT RESUME POINTS:
        {{selected_resume_text}}

        ### INSTRUCTION:
        {identity}
        Using ONLY the resume points provided in 'CANDIDATE MOST RELEVANT RESUME POINTS', write a professional and concise cold email tailored to the job.
        The email should:
        - Be personalized for the specific job description.
        - Highlight relevant experience, skills, and achievements from the selected points.
        - Be formal yet approachable.
        - Include a short and impactful introduction (1-2 lines).
        - End politely with interest in discussing the opportunity.

        Do not include a subject line or preamble. Only provide the email body.
        """

        prompt = PromptTemplate.from_template(prompt_template)
        chain_email = prompt | self.llm

        try:
            res = chain_email.invoke(
                {
                    "job_description": truncate_text_by_chars(str(job_description), max_chars=20000),
                    "selected_resume_text": selected_text,
                }
            )
            model_out = res.content
        except Exception:
            model_out = (
                "Dear Hiring Manager,\n\n"
                "I am interested in this role and my resume includes relevant experience. I would welcome a chance to discuss.\n\n"
                f"Best regards,\n{candidate_name if candidate_name else ''}"
            )

        # ensure signature uses extracted candidate_name when possible
        final_email = self._ensure_signature(model_out, candidate_name)
        return final_email
