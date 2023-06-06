from utilities import *
from pathlib import Path

# setup API key
openai_env_path, openai.api_key = None, None
cwd = Path.cwd()
openai_env_path = cwd / "openai.env"
set_open_ai_key(openai_env_path)

#Set up location for pdfs
pdf_uris = Path(cwd, "Prescribing-Info", "accupril_quinapril.pdf")

# actual resumes start on page 2 of this pdf compilation
drug1 = load_pdfs(pdf_uris)

create_index(drug1)
qa = create_conversation()
demo.queue(concurrency_count=3)
demo.launch()
