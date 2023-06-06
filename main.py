from utilities import *
from glob import glob

# setup API key
openai_env_path, openai.api_key = None, None
cwd = Path.cwd()
openai_env_path = ""
set_open_ai_key(openai_env_path)

#Set up location for pdfs
pdf_uris = Path(cwd, "Prescribing-Info", "accupril_quinapril.pdf")

# actual resumes start on page 2 of this pdf compilation
drug1 = load_resumes(pdf_uris)

create_index(drug1)
qa = create_conversation()
demo.queue(concurrency_count=3)
demo.launch()
