import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

# Import Chain: adjust this to match your file name (chains.py)
from chains import Chain

from utils import clean_text, load_resume_from_fileobj, load_resume_from_path

# Ensure WebBaseLoader has a USER_AGENT to avoid warnings
os.environ.setdefault("USER_AGENT", "jupyter-job-email-generator/0.1 (contact: your.email@example.com)")

st.set_page_config(layout="wide", page_title="Job Application Email Generator", page_icon="📧")

def create_streamlit_app(chain: Chain):
    st.title("📧 Personalized Job Application Email Generator")

    # Resume uploader
    st.sidebar.header("Upload Resume")
    uploaded_resume = st.sidebar.file_uploader(
        "Upload your resume (txt, pdf, docx) — txt recommended",
        type=["txt", "pdf", "docx"]
    )

    # Job input options (unchanged)
    st.header("Job Description Input")
    jd_input_type = st.radio("Provide job description via:", ("Paste JD text (recommended)", "Job posting URL", "Upload JD file"))
    job_text = ""
    job_url = ""

    if jd_input_type == "Paste JD text (recommended)":
        job_text = st.text_area("Paste the full job description here", height=250)
    elif jd_input_type == "Job posting URL":
        job_url = st.text_input("Enter job posting URL", value="")
    else:
        uploaded_jd = st.file_uploader("Upload a job description file", type=["txt", "pdf", "docx"], key="jd_upload")
        if uploaded_jd:
            try:
                # use the universal loader from utils to support pdf/docx/txt
                job_text = load_resume_from_fileobj(uploaded_jd)
            except Exception as e:
                st.warning(f"Could not parse uploaded JD file: {e}")
                job_text = ""

    # Generate button
    generate = st.button("Generate Email")

    if generate:
        # load resume text
        try:
            if uploaded_resume:
                resume_text = load_resume_from_fileobj(uploaded_resume)
            else:
                st.error("No resume provided. Upload one to generate an email.")
                return
        except Exception as e:
            st.error(f"Error loading resume: {e}")
            return

        # small preview so we can see extraction worked
        with st.expander("Resume preview (first 800 chars)"):
            st.write(resume_text[:800] + ("..." if len(resume_text) > 800 else ""))

        # assemble job text (from paste/upload/url)
        try:
            if jd_input_type == "Job posting URL":
                if not job_url:
                    st.error("Please enter a job posting URL.")
                    return
                loader = WebBaseLoader([job_url])
                docs = loader.load()
                if not docs:
                    st.error("Unable to scrape URL or no textual content found.")
                    return
                scraped = docs[0].page_content
                page_data = clean_text(scraped)
            else:
                if not job_text or job_text.strip() == "":
                    st.error("Please provide job description text (paste or upload).")
                    return
                page_data = clean_text(job_text)

            # Extract jobs and generate mails
            with st.spinner("Extracting job posting and generating emails..."):
                jobs = chain.extract_jobs(page_data)
                if not jobs:
                    st.warning("No job postings found in the provided text.")
                    return

                st.success(f"Found {len(jobs)} job posting(s). Generating emails...")
                for i, job in enumerate(jobs, start=1):
                    email = chain.write_personalized_mail(job, resume_text)
                    st.subheader(f"Email for job #{i}: {job.get('role', 'Unknown role')}")
                    st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    create_streamlit_app(chain)
