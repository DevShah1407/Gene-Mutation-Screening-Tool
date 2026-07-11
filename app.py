You are a senior Python, Streamlit, and software architecture engineer.

I generated a large Streamlit project (GeneVista AI) consisting of many files. The project architecture itself is complete, but I am unable to run it locally.

Please perform a COMPLETE debugging audit of the project.

These are the issues I encountered:

-------------------------------------------------------
1)

python app.py

returns

zsh: command not found: python

I am on macOS.

Python is installed as Python 3.14.3, so only python3 exists.

-------------------------------------------------------
2)

Running

python3 app.py

does NOT crash.

Instead it produces hundreds of warnings such as

missing ScriptRunContext

Session state does not function when running a script without `streamlit run`

Warning: to view this Streamlit app on a browser, run it with

streamlit run app.py

The app never opens properly.

-------------------------------------------------------

I want you to determine EVERYTHING that is wrong with the project.

Do NOT simply tell me to use "streamlit run".

Instead, inspect the entire codebase and determine whether there are deeper issues.

I want you to audit:

• application entrypoint
• imports
• package structure
• **init**.py files
• relative imports
• requirements.txt
• pyproject.toml
• Streamlit compatibility
• Python 3.14 compatibility
• session_state usage
• page configuration
• circular imports
• missing files
• missing assets
• missing directories
• configuration loading
• logging
• PDF generation
• Plotly usage
• document parser
• backend orchestrator
• all custom modules

Specifically verify whether every import used inside app.py actually exists.

Examples include:

from genevista.backend.analysis_orchestrator import AnalysisOrchestrator

from genevista.backend.document_parsing import DocumentParsingService

from genevista.reporting.pdf import PDFReportGenerator

from genevista.ui.components ...

etc.

If ANY imported file is missing,
incorrectly named,
contains syntax errors,
or imports unavailable libraries,
fix them.

Next:

Verify the project structure.

It should resemble something like

GeneVista/
│
├── app.py
├── requirements.txt
├── README.md
├── genevista/
│   ├── **init**.py
│   ├── backend/
│   ├── reporting/
│   ├── ui/
│   ├── domain/
│   ├── config.py
│   └── ...

If my structure is incorrect,
rewrite imports appropriately.

Next:

Check whether any packages are incompatible with Python 3.14.

If a dependency does not yet support Python 3.14,
replace it with a supported alternative or explain the required downgrade.

Next:

Verify Streamlit best practices.

Ensure

st.set_page_config()

is called exactly once.

Ensure

st.session_state

is used correctly.

Ensure no code executes before page configuration.

Ensure no blocking imports execute before Streamlit starts.

Next:

Check whether I accidentally generated backend code intended for FastAPI instead of Streamlit.

Next:

Verify that every file uploader,
Plotly chart,
PDF export,
logging configuration,
and CSS loader
works correctly.

Finally:

Produce a complete list of every issue found.

For each issue include:

1. Why it happens
2. Which file causes it
3. The exact code to replace
4. The corrected version
5. Why your fix works

If the project is already correct and the ONLY issue is that I launched the application incorrectly, explicitly state that.

Finally, provide the exact terminal commands I should execute on macOS from a clean terminal to run the application successfully.

Do not skip any files.

Treat this as a production code review for a professional biomedical AI application.
