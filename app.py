import streamlit as st
import pandas as pd
import openai
import io
import random

# This script (v15.1) contains the definitive fix for the tone of the opening
# sentence in low-scoring profiles, ensuring it remains neutral.

# --- Helper Function to convert DataFrame to Excel in memory ---
def to_excel(df):
    """Converts a pandas DataFrame to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Summaries')
    return output.getvalue()

# --- DEDICATED PROMPT TEMPLATES ---

def create_high_strength_prompt(salutation_name, pronoun, person_data, tied_competencies):
    """Creates the prompt for cases where the highest score is >= 3.5."""
    num_ties = len(tied_competencies)
    competency_list_str = ", ".join(tied_competencies)

    if num_ties == 1:
        example_instruction = f"HIGH-STRENGTH opening for: {competency_list_str}"
        example_sentence = f"{salutation_name} demonstrated a strong capacity to think strategically."
    elif num_ties == 2:
        example_instruction = f"HIGH-STRENGTH opening for: Strategic Thinker, Impactful Decision Maker"
        example_sentence = f"{salutation_name} evidenced a strong capacity to think strategically and make impactful decisions."
    else:
        example_instruction = f"HIGH-STRENGTH opening for: Results Driver, Strategic Thinker, Impactful Decision Maker"
        example_sentence = f"{salutation_name} demonstrated a strong ability to drive results, think strategically, and make impactful decisions."

    prompt_text = f"""
You are an elite talent management consultant and expert grammarian. Your task is to write an executive summary for {salutation_name}.

## NON-NEGOTIABLE CORE RULES
1.  **Language:** British English.
2.  **Structure:** A **single, unified paragraph**. You must not use any line breaks. This is a critical rule.
3.  **Competencies:** Describe behaviors, do not use the exact competency names as labels.
4.  **Tense:** All descriptions of observed, positive behaviors **MUST** be in the **past tense** (e.g., "He demonstrated...", "She showcased...").

## OPENING SENTENCE INSTRUCTION
Your first task is to construct a single, grammatically perfect opening sentence based on the instruction below.
* **Instruction:** `Create a 'HIGH-STRENGTH' opening for: {competency_list_str}`
* **Follow this specific format:** `[Name] [verb] a strong [capacity/ability] to [verb phrase(s)]`.
* **Reference Example:**
    * **Instruction:** `{example_instruction}`
    * **Correct Sentence:** `{example_sentence}`

## BODY OF SUMMARY INSTRUCTION
Beginning from the second sentence, address each competency from the input data below using the "Integrated Feedback Loop" structure: describe the positive behavior (in past tense), then immediately provide the related development area.

## Input Data for {salutation_name}
<InputData>
{person_data}
</InputData>

## FINAL INSTRUCTIONS
Create a strict single-paragraph summary between 250-280 words. Start with the grammatically perfect opening sentence you constructed, then follow the Integrated Feedback Loop for the body. Use {pronoun} after the first mention of {salutation_name}.
"""
    return prompt_text

def create_competence_prompt(salutation_name, pronoun, person_data, tied_competencies):
    """Creates the prompt for cases where the highest score is between 2.5 and 3.49."""
    num_ties = len(tied_competencies)
    competency_list_str = ", ".join(tied_competencies)

    if num_ties == 1:
        instruction_format = "the competence to [verb phrase]"
        example_instruction = f"COMPETENCE-SINGLE opening for: Talent Nurturer"
        example_sentence = f"{salutation_name} demonstrated the competence to nurture talent."
    else:
        instruction_format = "competence in [noun phrase(s)]"
        example_instruction = f"COMPETENCE-TIE opening for: Talent Nurturer, Strategic Thinker"
        example_sentence = f"{salutation_name} evidenced competence in talent nurturing and strategic thinking."

    prompt_text = f"""
You are an elite talent management consultant and expert grammarian. Your task is to write an executive summary for {salutation_name}.

## NON-NEGOTIABLE CORE RULES
1.  **Language:** British English.
2.  **Structure:** A **single, unified paragraph**. You must not use any line breaks. This is a critical rule.
3.  **Competencies:** Describe behaviors, do not use the exact competency names as labels.
4.  **Tense:** All descriptions of observed, positive behaviors **MUST** be in the **past tense** (e.g., "He demonstrated...", "She showcased...").

## OPENING SENTENCE INSTRUCTION
Your first task is to construct a single, grammatically perfect opening sentence based on the instruction below.
* **Instruction:** `Create a 'COMPETENCE' opening for: {competency_list_str}`
* **Follow this specific format:** `[Name] [verb] {instruction_format}`.
* **Reference Example:**
    * **Instruction:** `{example_instruction}`
    * **Correct Sentence:** `{example_sentence}`

## BODY OF SUMMARY INSTRUCTION
Beginning from the second sentence, address each competency from the input data below using the "Integrated Feedback Loop" structure: describe the positive behavior (in past tense), then immediately provide the related development area.

## Input Data for {salutation_name}
<InputData>
{person_data}
</InputData>

## FINAL INSTRUCTIONS
Create a strict single-paragraph summary between 250-280 words. Start with the grammatically perfect opening sentence you constructed, then follow with the Integrated Feedback Loop for the body. Use {pronoun} after the first mention of {salutation_name}.
"""
    return prompt_text

def create_developing_prompt(salutation_name, pronoun, person_data, tied_competencies):
    """Creates the prompt for cases where the highest score is less than 2.5."""
    # This prompt now has the final rule for neutral tone.
    prompt_text = f"""
You are an elite talent management consultant and expert grammarian. Your task is to write an executive summary for {salutation_name}.

## NON-NEGOTIABLE CORE RULES
1.  **Language:** British English.
2.  **Structure:** A **single, unified paragraph**. You must not use any line breaks. This is a critical rule.
3.  **Competencies:** Describe behaviors, do not use the exact competency names as labels.
4.  **Tense:** All descriptions of observed, positive behaviors **MUST** be in the **past tense** (e.g., "He demonstrated...", "She showcased...").

## WRITING INSTRUCTION
* **Opening Sentence:** You **MUST NOT** use a formulaic opening sentence for this summary. You must begin the summary **immediately** by describing the most positive observed behavior from the highest-scoring competency in a narrative style.
* **CRITICAL TONE RULE:** This opening statement **MUST be neutral**. Do not use any positive adjectives like "strong," "notable," or "exceptional."
* **Body of Summary:** After the neutral opening, you must continue by addressing all other competencies using the "Integrated Feedback Loop" structure: describe the positive behavior for a competency (in past tense), then immediately provide the related development area.
* **Study the following example closely and replicate its neutral style and structure:**
    * **Example for a low-scoring profile:** "Fatema evidenced collaboration by engaging positively with different parties, internally and externally, offering support and ensuring shared goals were properly achieved. To further develop her skills, Fatema may need to strengthen her self-confidence and the ability to communicate clearly... She demonstrated customer advocacy through resolving customers’ issues... Fatema may benefit from identifying customers’ needs..."

## Input Data for {salutation_name}
<InputData>
{person_data}
</InputData>

## FINAL INSTRUCTIONS
Create a strict single-paragraph summary between 250-280 words. Begin narratively with a neutral tone as instructed and follow the Integrated Feedback Loop for the entire body. Use {pronoun} after the first mention of {salutation_name}.
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
                {"role": "system", "content": "You are an elite talent management consultant and expert grammarian. You follow all instructions with absolute precision, especially the non-negotiable core rules about structure and tense."},
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

# --- Main App Logic ---
def select_and_create_prompt(salutation_name, pronoun, person_data, scores_dict):
    """Determines the scenario and calls the appropriate prompt creation function."""
    if not scores_dict: return ""
    
    highest_score = max(scores_dict.values())
    tied_competencies = [comp for comp, score in scores_dict.items() if score == highest_score]
    
    # Rule A: >= 3.5
    if highest_score >= 3.5:
        return create_high_strength_prompt(salutation_name, pronoun, person_data, tied_competencies)

    # Rule B: >= 2.5 and < 3.5
    elif highest_score >= 2.5:
        return create_competence_prompt(salutation_name, pronoun, person_data, tied_competencies)

    # Rule C: < 2.5
    else:
        return create_developing_prompt(salutation_name, pronoun, person_data, tied_competencies)

# --- Streamlit App Main UI ---
st.set_page_config(page_title="DGE Executive Summary Generator v15.1", layout="wide")
st.title("📄 DGE Executive Summary Generator (V15.1)")
st.markdown("""
This application generates professional executive summaries based on leadership competency scores.
**Version 15.1 contains the definitive fix for all opening sentence tone and grammar.**
1.  **Set up your secrets**.
2.  **Download the Sample Template**.
3.  **Upload your completed Excel file**.
4.  **Click 'Generate Summaries'**.
""")

# --- Create and provide a sample file for download ---
sample_data = {
    'email': ['bechara.b@example.com', 'hapless.h@example.com'],
    'salutation_name': ['Bechara', 'Hapless'],
    'gender': ['M', 'M'],
    'level': ['Manager', 'Specialist'],
    'Strategic Thinker': [2.05, 3.1],
    'Impactful Decision Maker': [1.62, 3.1],
    'Effective Collaborator': [2.47, 3.04],
    'Talent Nurturer': [1.97, 3.3],
    'Results Driver': [2.03, 3],
    'Customer Advocate': [2.47, 2.9],
    'Transformation Enabler': [2.01, 3],
    'Innovation Explorer': [2.31, 3]
}
sample_df = pd.DataFrame(sample_data)
sample_excel_data = to_excel(sample_df)

st.download_button(
    label="📥 Download Sample Template File (V15.1)",
    data=sample_excel_data,
    file_name="dge_summary_template_v15.1.xlsx",
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
            
            all_known_competencies = [
                'Strategic Thinker', 'Impactful Decision Maker', 'Effective Collaborator',
                'Talent Nurturer', 'Results Driver', 'Customer Advocate',
                'Transformation Enabler', 'Innovation Explorer'
            ]
            
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

                # Select the correct, tailored prompt for the specific scenario
                prompt = select_and_create_prompt(salutation_name, pronoun, person_data_str, scores_dict)
                
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
                st.subheader("Generated Summaries (V15.1)")
                
                output_df = df.copy()
                output_df['Executive Summary'] = generated_summaries
                
                st.dataframe(output_df)
                
                results_excel_data = to_excel(output_df)
                st.download_button(
                    label="📥 Download V15.1 Results as Excel",
                    data=results_excel_data,
                    file_name="Generated_Executive_Summaries_V15.1.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
