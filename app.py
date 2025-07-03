import streamlit as st
import pandas as pd
import openai
import io
import random

# This script (v9.1) is the final version with a grammatical fix in the
# programmatic sentence builder to ensure all openings are correct.

# --- Helper Function to convert DataFrame to Excel in memory ---
def to_excel(df):
    """Converts a pandas DataFrame to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Summaries')
    return output.getvalue()

# --- Mappings and Helper Functions for Sentence Construction ---

# Maps competency names to their VERB phrase representation.
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

# NEW: Maps competency names to their NOUN phrase representation for grammatical correctness.
COMPETENCY_TO_NOUN_PHRASE = {
    'Strategic Thinker': 'strategic thinking',
    'Impactful Decision Maker': 'impactful decision-making',
    'Effective Collaborator': 'collaboration',
    'Talent Nurturer': 'talent nurturing',
    'Results Driver': 'a drive for results',
    'Customer Advocate': 'customer advocacy',
    'Transformation Enabler': 'transformation enablement',
    'Innovation Explorer': 'innovation exploration'
}

def format_list_for_sentence(item_list):
    """Formats a list of strings into a natural language string with commas and 'and'."""
    if not item_list:
        return ""
    if len(item_list) == 1:
        return item_list[0]
    if len(item_list) == 2:
        return f"{item_list[0]} and {item_list[1]}"
    # For 3 or more items, use the Oxford comma style.
    return ", ".join(item_list[:-1]) + f", and {item_list[-1]}"

def determine_opening_sentence(salutation_name, scores_dict):
    """
    Programmatically determines the precise opening sentence based on the highest score
    and handles all tie-breaker scenarios with correct grammar.
    """
    if not scores_dict:
        return ""

    highest_score = max(scores_dict.values())
    tied_competencies = [comp for comp, score in scores_dict.items() if score == highest_score]
    
    verb = random.choice(["evidenced", "demonstrated"])

    # Rule A: Highest score is 3.5 or greater
    if highest_score >= 3.5:
        verb_phrases = [COMPETENCY_TO_VERB_PHRASE.get(c, c.lower()) for c in tied_competencies]
        formatted_competencies = format_list_for_sentence(verb_phrases)
        capacity_or_ability = random.choice(["a strong capacity to", "a strong ability to"])
        return f"{salutation_name} {verb} {capacity_or_ability} {formatted_competencies}."

    # Rule B: Highest score is between 2.5 and 3.49
    elif highest_score >= 2.5:
        verb_phrases = [COMPETENCY_TO_VERB_PHRASE.get(c, c.lower()) for c in tied_competencies]
        formatted_competencies = format_list_for_sentence(verb_phrases)
        if len(tied_competencies) > 1:
             return f"{salutation_name} {verb} competence in {formatted_competencies}."
        else:
             return f"{salutation_name} {verb} the competence to {formatted_competencies}."

    # Rule C: Highest score is less than 2.5
    else:
        # GRAMMAR FIX: Use noun phrases for this case to be grammatically correct.
        noun_phrases = [COMPETENCY_TO_NOUN_PHRASE.get(c, c.lower()) for c in tied_competencies]
        formatted_nouns = format_list_for_sentence(noun_phrases)
        return f"{salutation_name} {verb} {formatted_nouns}."


# --- The RE-ENGINEERED Master Prompt Template (Version 9.1) ---
def create_master_prompt(salutation_name, pronoun, person_data, opening_sentence):
    """
    Dynamically creates the prompt. The opening sentence is now pre-built and passed in.
    """
    prompt_text = f"""
You are an elite talent management consultant from a top-tier firm. Your writing is strategic, cohesive, and you follow instructions with absolute precision.

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

1.  **CRITICAL OPENING SENTENCE (MANDATORY):**
    * The summary **MUST** begin with the following sentence, exactly as it is written.
    * **DO NOT** alter, add to, or deviate from this sentence in any way.
    * **Opening Sentence:** "{opening_sentence}"

2.  **STRUCTURE AFTER OPENING: The Integrated Feedback Loop.**
    * Beginning from the **second sentence**, the rest of the paragraph **MUST** address each competency one by one in a logical flow.
    * For each competency, you will first describe the **observed positive behavior**.
    * Then, **IMMEDIATELY AFTER** describing the behavior, you will provide the **related development area** for that same competency, introduced with a phrase like "As a next step...", "To build on this...", or "He may benefit from...".
    * This structure of `[Strength 1] -> [Development for 1] -> [Strength 2] -> [Development for 2]` is mandatory for the body of the paragraph.

3.  **Name and Pronoun Usage:**
    * The candidate's name is already in the opening sentence. After that, use only the pronoun **{pronoun}**.

## ---------------------------------------------
## ANALYSIS OF A GOLD-STANDARD EXAMPLE (INTERNALIZE THE NEW LOGIC)
## ---------------------------------------------

**This example demonstrates the required structure: A pre-defined opening sentence, followed by the Integrated Feedback Loop, all within a SINGLE PARAGRAPH.**

* **Correct Output Example:** "Khasiba demonstrated strong ability to drive results, think strategically, and make impactful decisions. She consistently showcased a capacity to align initiatives with organisational objectives, maintain focus on key deliverables, and adapt her approach to achieve outcomes efficiently. To further develop this area, she could refine her ability to manage competing priorities in high-pressure situations. Her strategic thinking was evident in her ability to assess complex scenarios and anticipate broader implications. As a next step, she could focus on enhancing her ability to identify emerging trends and integrate them into actionable strategies. She displayed thoughtful decision-making by balancing pragmatism with insight. To build on this strength, she could work towards strengthening her ability to navigate ambiguity..."

* **Analysis of the Integrated Logic:**
    * **Strict Opening:** The summary begins with the exact sentence provided to it.
    * **Cohesion:** After the opening, the summary flows logically through the other competencies.
    * **Integrated Feedback:** The development point for a competency comes *immediately* after its description.

## ---------------------------------------------
## FINAL INSTRUCTIONS
## ---------------------------------------------

Now, process the data for {salutation_name}. Create a **strict single-paragraph summary**. Start with the provided opening sentence. The rest of the paragraph must follow the **Integrated Feedback Loop** structure. The total word count should remain between 250-280 words.
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
                {"role": "system", "content": "You are an elite talent management consultant. Your writing is strategic and cohesive. You follow all instructions with absolute precision, especially using the provided opening sentence verbatim."},
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
st.set_page_config(page_title="DGE Executive Summary Generator v9.1", layout="wide")
st.title("📄 DGE Executive Summary Generator (V9.1)")
st.markdown("""
This application generates professional executive summaries based on leadership competency scores.
**Version 9.1 includes the final grammar fix for opening sentences.**
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
    label="📥 Download Sample Template File (V9.1)",
    data=sample_excel_data,
    file_name="dge_summary_template_v9.1.xlsx",
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

                # Programmatically build the opening sentence
                opening_sentence = determine_opening_sentence(salutation_name, scores_dict)

                # Create the prompt with the pre-built sentence
                prompt = create_master_prompt(salutation_name, pronoun, person_data_str, opening_sentence)
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
                st.subheader("Generated Summaries (V9.1)")
                
                output_df = df.copy()
                output_df['Executive Summary'] = generated_summaries
                
                st.dataframe(output_df)
                
                results_excel_data = to_excel(output_df)
                st.download_button(
                    label="📥 Download V9.1 Results as Excel",
                    data=results_excel_data,
                    file_name="Generated_Executive_Summaries_V9.1.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
