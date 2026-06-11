## Run it yourself

**Prerequisites:** [Python 3.11+](https://www.python.org/downloads/), [Ollama](https://ollama.com), and [Git](https://git-scm.com/).

```bash
# 1. Clone the repo
git clone https://github.com/Navigatedhustle/local-health-intake-ai.git
cd local-health-intake-ai

# 2. Pull the local model (about 7GB, one time)
ollama pull mistral-nemo

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Your browser opens automatically at `localhost:8501`. Make sure the Ollama app is running before you launch.

---

## What it captures

The agent extracts and validates the following structured fields from a natural conversation:

- **Chief complaint** in the patient's own words
- **Duration** of the symptom
- **Severity score** (1 to 10)
- **Additional symptoms**
- **Allergies and current medications**
- **Red flags** (chest pain, difficulty breathing, severe pain, and similar)
- **Suggested triage urgency** (Routine, Urgent, Emergency) with reasoning

Output is enforced as a Pydantic-validated schema, which means near-total format reliability instead of free-text the care team has to clean up.

---

## Important disclaimers

This is a demonstration project. It uses synthetic patient input only. It is **not a medical device**, does **not diagnose or treat**, and does **not replace evaluation by a licensed healthcare provider**. Any real clinical deployment would require formal validation, clinical oversight, and regulatory review.

The point being demonstrated is architectural: that capable clinical AI can run entirely on local compute, eliminating the third-party data transmission that creates most healthcare AI privacy risk.

---

## License

MIT. See [LICENSE](LICENSE) if present, or use freely with attribution.
