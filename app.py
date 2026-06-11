import streamlit as st
import intake_agent
import report

st.set_page_config(
    page_title="Local AI Clinical Intake",
    page_icon="+",
    layout="centered",
)

st.markdown("""
<style>
.privacy-banner {
    background-color: #E1F5EE;
    border: 1px solid #1D9E75;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    color: #0F6E56;
    font-size: 14px;
    font-weight: 500;
    text-align: center;
}
.main-title {
    font-size: 28px;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 4px;
}
.subtitle {
    font-size: 15px;
    color: #666;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Local AI Clinical Intake Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Pre-visit symptom intake, powered entirely by on-device AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="privacy-banner">No data transmitted - 100% local processing - HIPAA-safe by design</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("About this demo")
    st.write(
        "This intake agent runs a 12-billion parameter language model "
        "**entirely on local hardware**. No patient data ever leaves the device."
    )
    st.divider()
    st.subheader("Running on")
    st.write("- AMD Ryzen 7 9800X3D")
    st.write("- NVIDIA RTX 5070 Ti (16GB VRAM)")
    st.write("- Model: Mistral-Nemo via Ollama")
    st.divider()
    st.caption(
        "Demonstration only. Not a medical device. Uses synthetic patient input. "
        "Does not diagnose or treat."
    )
    st.divider()
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.spinner("Starting local intake assistant..."):
        opening = intake_agent.get_opening_message()
    st.session_state.messages.append({"role": "assistant", "content": opening})

if "report" not in st.session_state:
    st.session_state.report = None

for msg in st.session_state.messages:
    role = "assistant" if msg["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.write(msg["content"])

user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally..."):
            ai_reply = intake_agent.get_ai_response(st.session_state.messages)
        st.write(ai_reply)

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    generate_clicked = st.button("Generate Intake Report", type="primary", use_container_width=True)

if generate_clicked:
    if len(st.session_state.messages) < 3:
        st.warning("Please have a short conversation first so the assistant can gather your symptoms.")
    else:
        with st.spinner("Extracting structured intake data locally..."):
            extracted = intake_agent.extract_structured_data(st.session_state.messages)

        if extracted is None:
            st.error("Could not extract structured data. Try answering a few more questions, then generate again.")
        else:
            st.session_state.report = extracted

if st.session_state.report is not None:
    rep = st.session_state.report
    st.success("Intake report generated locally.")

    st.subheader("Structured Intake Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Severity", f"{rep.severity_score}/10")
    c2.metric("Urgency", rep.suggested_urgency)
    c3.metric("Red Flags", len(rep.red_flags))

    st.write("**Chief Complaint:**", rep.chief_complaint)
    st.write("**Duration:**", rep.duration)

    if rep.additional_symptoms:
        st.write("**Additional Symptoms:**", ", ".join(rep.additional_symptoms))

    st.write("**Allergies:**", rep.allergies)
    st.write("**Current Medications:**", rep.current_medications)

    if rep.red_flags:
        st.error("Red flags detected: " + ", ".join(rep.red_flags))

    st.info(f"Triage reasoning: {rep.urgency_reasoning}")

    with st.expander("View raw JSON output"):
        st.json(rep.model_dump())

    pdf_bytes = report.generate_pdf(rep)
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="clinical_intake_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )