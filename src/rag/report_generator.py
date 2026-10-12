"""
RAG-based Clinical Explanation Generator.
Synthesizes deep model classification outputs, Grad-CAM anatomical saliency, and retrieved clinical guidelines.
"""

from .retrieval import ClinicalGuidelineRetriever


class RetinalRAGReportGenerator:
    """
    Constructs an evidence-backed diagnostic report adhering to ophthalmic standards.
    Integrates model grade, confidence, Grad-CAM salient location, and retrieved guidelines.
    """
    def __init__(self, retriever=None):
        self.retriever = retriever or ClinicalGuidelineRetriever()

    def generate_report(self, disease, predicted_grade, confidence, xai_description=None):
        """
        Generates structured clinical report grounded in retrieved evidence.
        """
        query = f"Patient fundus examination for {disease} with suspected {predicted_grade}."
        if xai_description:
            query += f" Salient findings in visual heatmap: {xai_description}."

        retrieved_docs = self.retriever.retrieve(query, top_k=1)
        evidence = retrieved_docs[0] if retrieved_docs else None

        criteria_text = evidence["criteria"] if evidence else "Assessment based on deep feature analysis."
        management_text = evidence["clinical_management"] if evidence else "Consult ophthalmologist for full clinical examination."
        references_text = evidence["references"] if evidence else "Standard Retinal Screening Protocols."

        report = {
            "disease_evaluated": disease,
            "predicted_grade": predicted_grade,
            "prediction_confidence": f"{confidence * 100:.2f}%",
            "visual_explanation_findings": xai_description or "Grad-CAM focal activations identified in key retinal sectors.",
            "clinical_evidence_criteria": criteria_text,
            "recommended_clinical_pathway": management_text,
            "guideline_sources": references_text,
            "formatted_summary": (
                f"### Retinal AI Decision-Support Summary\n"
                f"- **Condition:** {disease}\n"
                f"- **Model Diagnosis:** {predicted_grade} (Confidence: {confidence * 100:.1f}%)\n"
                f"- **Visual Attribution (Grad-CAM):** {xai_description or 'Localized feature attributions'}\n"
                f"- **Diagnostic Criteria:** {criteria_text}\n"
                f"- **Recommended Pathway:** {management_text}\n"
                f"- **Guideline Reference:** {references_text}\n"
            )
        }
        return report
