import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_rule_based_explanation(data):
    """Fallback explanation generator based on fixed medical rules."""
    reasons = []

    if data.get("age", 0) > 60:
        reasons.append("advanced age (>60)")
    if data.get("avg_glucose_level", 0) > 140:
        reasons.append("elevated blood glucose levels")
    if data.get("bmi", 0) > 30:
        reasons.append("high Body Mass Index (BMI > 30)")
    if data.get("heart_disease") == 1:
        reasons.append("pre-existing heart disease condition")
    if data.get("hypertension") == 1:
        reasons.append("history of hypertension (high blood pressure)")
    if data.get("smoking_status") == 2:
        reasons.append("active smoking status")

    if reasons:
        return "The prediction is primarily influenced by: " + ", ".join(reasons) + "."
    return "The patient indicators don't strongly align with typical high-risk stroke factors."

def explain_with_llm(patient_data, probability, risk, shap_contributors=None):
    """
    Generates a personalized medical explanation using OpenAI LLM.
    Falls back to rule-based explanation if the API call fails.
    """
    
    # Constructing the context for the LLM
    extra = patient_data.get("extra_metrics", {})
    extra_str = ""
    if extra:
        extra_str = "\nAdditional Lab Metrics:\n"
        for k, v in extra.items():
            extra_str += f"- {k.replace('_', ' ').title()}: {v}\n"

    context = (
        f"Patient Profile:\n"
        f"- Age: {patient_data.get('age')}\n"
        f"- BMI: {patient_data.get('bmi')}\n"
        f"- Average Glucose Level: {patient_data.get('avg_glucose_level')}\n"
        f"- Hypertension: {'Yes' if patient_data.get('hypertension') == 1 else 'No'}\n"
        f"- Heart Disease: {'Yes' if patient_data.get('heart_disease') == 1 else 'No'}\n"
        f"- Smoking Status: {patient_data.get('smoking_status')}\n"
        f"{extra_str}\n"
        f"Model Prediction:\n"
        f"- Risk Level: {risk}\n"
        f"- Probability: {round(probability, 2)}%\n"
    )

    if shap_contributors:
        context += f"\nKey Feature Contributions (Predictive Model): {', '.join(shap_contributors)}\n"

    prompt = (
        f"You are a specialized medical clinical assistant. Based on the following data, provide a professional, "
        f"empathetic, and easy-to-understand explanation for the patient's stroke risk. "
        f"Focus on the most significant risk factors identified and suggest general healthy lifestyle changes. "
        f"IMPORTANT: State that this is a system-generated assessment and not a definitive medical diagnosis.\n\n"
        f"{context}\n\n"
        f"Explanation:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o for medical context
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant specializing in stroke risk analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"LLM API Error: {e}")
        # Fallback to rule-based explanation
        return generate_rule_based_explanation(patient_data)

if __name__ == "__main__":
    # Test block
    test_patient = {
        "age": 67,
        "bmi": 36.6,
        "avg_glucose_level": 228.69,
        "hypertension": 0,
        "heart_disease": 1,
        "smoking_status": "formerly smoked"
    }
    print(explain_with_llm(test_patient, 85.5, "High Stroke Risk", ["Age (+)", "Glucose (+)", "Heart Disease (+)"]))
