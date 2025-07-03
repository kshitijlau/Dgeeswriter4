import streamlit as st
import pandas as pd
import openai
import io
import random

# This script (v10.0) reverts the opening sentence creation back to the AI,
# but with a new, highly detailed set of prompt instructions to ensure grammatical correctness.

# --- Helper Function to convert DataFrame to Excel in memory ---
def to_excel(df):
    """Converts a pandas DataFrame to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Summaries')
    return output.getvalue()

# --- Mappings for Prompt Examples ---
# This is now only used to help construct examples within the prompt itself.
COMPETENCY_TO_VERB_PHRASE = {
    'Strategic Thinker': 'think strategically',
    'Impactful Decision Maker': 'make impactful decisions',
    'Effective Collaborator': 'collaborate effectively',
    'Talent Nurturer': 'nurture talent',
    'Results Driver': 'drive results',
    'Customer Advocate': 'advocate for customers',
    'Transformation Enabler': 'enable transformation',
    'Innovation Explorer': 'explore innovation'
}

# --- The RE-ENGINEERED Master Prompt Template (Version 10.0) ---
def create_master_prompt(salutation_name, pronoun, person_data):
    """
    Dynamically creates the prompt. The AI is now responsible for creating the
    opening sentence based on a new, highly detailed set of rules.
    """
    prompt_text = f"""
You are an elite talent management consultant from a top-tier firm. Your writing is strategic, cohesive, and you follow instructions with absolute precision and perfect grammar.

## NON-NEGOTIABLE CORE RULES
1.  **Language:** The entire summary MUST be written in **British English**.
2.  **Structure:** The entire summary MUST be a **single, unified paragraph**. There can be no deviation from this rule.
3.  **Competencies:** You MUST NOT use the exact name of a competency (e.g., "Results Driver") in the narrative. Instead, you MUST describe the behavior using a verb phrase (e.g., "...demonstrated an ability to drive results," or "...showcased strategic thinking.").

## Core Objective
Synthesize the provided competency data for {salutation_name} into a single, cohesive, and integrated narrative paragraph that adheres to all core rules.

## Input Data for {salutation_name}
<InputData>
{person_data}
</InputData>

## ---------------------------------------------
## CRITICAL DIRECTIVES FOR SUMMARY STRUCTURE & TONE
## ---------------------------------------------

1.  **CRITICAL OPENING SENTENCE PROTOCOL (AI-DRIVEN CONDITIONAL LOGIC):**
    * Your first task is to analyze the input data to find the single highest numerical score.
    * You will then construct the opening sentence based on the score's value and whether there is a tie, following these rules precisely.

    * **Rule A: If the highest score is 3.5 or greater (>= 3.5):**
        * The sentence must reflect a clear strength using the format: `[Name] [verb] a strong [capacity/ability] to [verb phrase(s)]`.
        * **For a single highest competency:** "Osman demonstrated a strong capacity to think strategically."
        * **For a 2-way tie:** "Osman evidenced a strong capacity to think strategically and make impactful decisions."
        * **For a 3-way tie:** "Khasiba demonstrated a strong ability to drive results, think strategically, and make impactful decisions."

    * **Rule B: If the highest score is between 2.5 and 3.49 (inclusive):**
        * The sentence must reflect competence.
        * **For a single highest competency:** "Pitiable demonstrated the competence to nurture talent." (Structure: `...the competence to [verb phrase]`)
        * **For a 2-way tie or more:** "Random evidenced competence in nurturing talent and thinking strategically." (Structure: `...competence in [noun phrase(s)]`)

    * **Rule C: If the highest score is less than 2.5 (< 2.5):**
        * The sentence must be a neutral observation of the highest-scoring behaviors.
        * **For a single highest competency:** "Fatema evidenced collaboration."
        * **For a 2-way tie or more:** "Wretched evidenced collaboration and customer advocacy." (Structure: `...[verb] [noun phrase(s)]`)

2.  **STRUCTURE AFTER OPENING: The Integrated Feedback Loop.**
    * Beginning from the **second sentence**, the rest of the paragraph **MUST** address each competency one by one in a logical flow.
    * For each competency, you will first describe the **observed positive behavior**.
    * Then, **IMMEDIATELY AFTER** describing the behavior, you will provide the **related development area** for that same competency, introduced with a phrase like "As a next step...", "To build on this...", or "He may benefit from...".

3.  **Name and Pronoun Usage:**
    * Use the candidate's name in the opening sentence. After that, use only the pronoun **{pronoun}**.

## ---------------------------------------------
## FINAL INSTRUCTIONS
## ---------------------------------------------

Now, process the data for {salutation_name}. First, analyze the scores to determine the correct opening sentence structure (A, B, or C) and ensure it is grammatically perfect. Then, create a **strict single-paragraph summary** that follows the **Integrated Feedback Loop** structure. The total word count should remain between 250-280 words.
"""
    return prompt_text

# --- API Call Function for Azure OpenAI ---
def generate_summary_azure(prompt, api_key, endpoint, deployment_name):
    """Calls the Azure OpenAI API to generate a summary."""
    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-01"
        )
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an elite talent management consultant and expert grammarian. You follow all instructions with absolute precision, especially the conditional logic for constructing a grammatically perfect opening sentence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while contacting Azure OpenAI: {e}")
        return None

# --- Streamlit App Main UI ---
st.set_page_config(page_title="DGE Executive Summary Generator v10.0", layout="wide")
st.title("📄 DGE Executive Summary Generator (V10.0)")
st.markdown("""
This application generates professional executive summaries based on leadership competency scores.
**Version 10.0 uses an advanced AI-driven prompt to ensure grammatical correctness.**
1.  **Set up your secrets**.
2.  **Download the Sample Template**.
3.  **Upload your completed Excel file**.
4.  **Click 'Generate Summaries'**.
""")

# --- Create and provide a sample file for download ---
sample_data = {
    'email': ['wretched.w@example.com', 'pitiable.p@example.com', 'bechari.b@example.com', 'bechara.b@example.com'],
    'salutation_name': ['Wretched', 'Pitiable', 'Bechari', 'Bechara'],
    'gender': ['M', 'F', 'F', 'M'],
    'level': ['Specialist', 'Manager', 'Specialist', 'Manager'],
    'Strategic Thinker': [2.05, 3.1, 2.09, 2.05],
    'Impactful Decision Maker': [1.62, 3.1, 2.09, 1.62],
    'Effective Collaborator': [2.47, 3.04, 2.09, 2.47],
    'Talent Nurturer': [1.97, 3.3, 2.06, 1.97],
    'Results Driver': [2.03, 3, 2.05, 2.03],
    'Customer Advocate': [2.47, 2.9, 2.03, 2.47],
    'Transformation Enabler': [2.01, 3, 2.01, 2.01],
    'Innovation Explorer': [2.31, 3, 2, 2.31]
}
sample_df = pd.DataFrame(sample_data)
sample_excel_data = to_excel(sample_df)

st.download_button(
    label="📥 Download Sample Template File (V10.0)",
    data=sample_excel_data,
    file_name="dge_summary_template_v10.0.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.divider()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your completed Excel file here", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Excel file loaded successfully. Ready to generate summaries.")
        st.dataframe(df.head())

        if st.button("Generate Summaries", key="generate"):
            try:
                azure_api_key = st.secrets["azure_openai"]["api_key"]
                azure_endpoint = st.secrets["azure_openai"]["endpoint"]
                azure_deployment_name = st.secrets["azure_openai"]["deployment_name"]
            except (KeyError, FileNotFoundError):
                st.error("Azure OpenAI credentials not found. Please configure them in your Streamlit secrets.")
                st.stop()
            
            all_known_competencies = list(COMPETENCY_TO_VERB_PHRASE.keys())
            
            if 'salutation_name' not in df.columns:
                st.error("Error: The uploaded file is missing the required 'salutation_name' column.")
                st.stop()

            generated_summaries = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                salutation_name = row['salutation_name']
                gender_input = str(row.get('gender', '')).upper()
                pronoun = 'They'
                if gender_input == 'M': pronoun = 'He'
                elif gender_input == 'F': pronoun = 'She'
                
                st.write(f"Processing summary for: {salutation_name}...")
                
                scores_dict = {comp: float(row[comp]) for comp in all_known_competencies if comp in row and pd.notna(row[comp])}
                person_data_str = "\n".join([f"- {comp}: {score}" for comp, score in scores_dict.items()])

                # The prompt now handles all logic internally.
                prompt = create_master_prompt(salutation_name, pronoun, person_data_str)
                summary = generate_summary_azure(prompt, azure_api_key, azure_endpoint, azure_deployment_name)
                
                if summary:
                    generated_summaries.append(summary)
                    st.success(f"Successfully generated summary for {salutation_name}.")
                else:
                    generated_summaries.append("Error: Failed to generate summary.")
                    st.error(f"Failed to generate summary for {salutation_name}.")

                progress_bar.progress((i + 1) / len(df))

            if generated_summaries:
                st.balloons()
                st.subheader("Generated Summaries (V10.0)")
                
                output_df = df.copy()
                output_df['Executive Summary'] = generated_summaries
                
                st.dataframe(output_df)
                
                results_excel_data = to_excel(output_df)
                st.download_button(
                    label="📥 Download V10.0 Results as Excel",
                    data=results_excel_data,
                    file_name="Generated_Executive_Summaries_V10.0.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
