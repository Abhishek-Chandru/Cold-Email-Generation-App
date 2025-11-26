import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found in environment. Set it in a .env file or environment variables."
            )

        # Use a currently supported Groq model
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",  # <- updated model name
        )

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
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            parsed = json_parser.parse(res.content)
        except OutputParserException:
            # bubble a helpful error
            raise OutputParserException(
                "Failed to parse LLM output to JSON. Output may be too large or not valid JSON."
            )

        # ensure list
        return parsed if isinstance(parsed, list) else [parsed]

    def write_personalized_mail(self, job, resume_text, applicant_name: str = "Name",
                                applicant_title: str = ""):
        # normalize job description text
        job_description = job if isinstance(job, str) else (job.get("description") or job.get("role") or str(job))

        # If you already do selection of relevant resume points elsewhere, keep that.
        # For simplicity this example passes full resume (or you can pass selected_text)
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### CANDIDATE RESUME (RELEVANT POINTS):
            {resume_text}

            ### INSTRUCTION:
            You are {applicant_name}, a motivated and skilled candidate applying for the job above.
            Using the resume details provided, write a professional and concise cold email to apply for this role.
            The email should:
            - Be personalized for the specific job description.
            - Highlight relevant experience, skills, and achievements from the resume.
            - Be formal yet approachable.
            - Include a short and impactful introduction (1-2 lines).
            - End politely with interest in discussing the opportunity and a closing signature using the supplied applicant name and title.

            Do not include a subject line or preamble. Only provide the email body.
            ### EMAIL:
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": str(job_description),
            "resume_text": resume_text,
            "applicant_name": applicant_name,
            "applicant_title": applicant_title or ""
        })

        # Optionally append the sign-off if model doesn't include it:
        email_body = res.content
        if applicant_title and applicant_name and applicant_name.lower() not in email_body.lower():
            # ensure sign-off exists
            email_body = email_body.strip() + f"\n\nBest regards,\n{applicant_name}\n{applicant_title}"
        elif applicant_name and applicant_name.lower() not in email_body.lower():
            email_body = email_body.strip() + f"\n\nBest regards,\n{applicant_name}"

        return email_body
