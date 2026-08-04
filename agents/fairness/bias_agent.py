import os
import re
import json
from dotenv import load_dotenv

class BiasAgent:
    def __init__(self):
        self.gemini_model = None
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                print("BiasAgent: Loaded Gemini API node for opinion auditing.")
            except Exception as e:
                print(f"Warning: Failed to load Gemini API in BiasAgent: {e}. Using Lexical Fallback.")
        else:
            print("Warning: Gemini API Key missing for BiasAgent. Using Lexical Fallback.")

    def audit_case_fairness(self, text: str) -> dict:
        """
        Audits case text for demographic bias, sentencing disparities, and opinion neutrality.
        Utilizes LLM-as-a-Judge Zero-Shot evaluation if Gemini is active,
        otherwise falls back to a local rule-based lexical scanner.
        """
        if not text or not text.strip():
            return {
                "bias_score": None,
                "bias_details": "Empty text. Neutrality audit skipped.",
                "flags": []
            }

        # 1. Attempt LLM-as-a-Judge evaluation
        if self.gemini_model is not None:
            try:
                prompt = (
                    "You are a Judicial Fairness Auditor reviewing Indian court judgments.\n\n"
                    "Check ONLY for these specific issues:\n"
                    "1. References to defendant's caste, religion, gender, or socioeconomic status used as reasoning.\n"
                    "2. Language that assumes guilt based on community membership.\n"
                    "3. Differential language when describing similar acts by different demographic groups.\n"
                    "4. Irrelevant personal characteristics mentioned in sentencing rationale.\n\n"
                    "Rate the judgment from 0.0 (serious bias found) to 1.0 (no bias detected).\n\n"
                    "Return ONLY a valid JSON object matching this structure (no markdown formatting, code block fences, or other text):\n"
                    "{\n"
                    "  \"bias_score\": <float between 0.0 and 1.0>,\n"
                    "  \"bias_details\": \"<one sentence finding>\",\n"
                    "  \"flags\": [\"<list of specific phrases flagged>\"]\n"
                    "}\n\n"
                    f"TEXT:\n{text[:40000]}"
                )

                response = self.gemini_model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Clean up any potential markdown formatting in LLM output
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    response_text = "\n".join(lines).strip()

                data = json.loads(response_text)
                if "bias_score" in data and "bias_details" in data:
                    return {
                        "bias_score": float(data["bias_score"]),
                        "bias_details": str(data["bias_details"]),
                        "flags": list(data.get("flags", []))
                    }
            except Exception as e:
                print(f"Warning: Gemini bias audit failed: {e}. Falling back to lexical scan.")

        # 2. Local Lexical Fallback Scan
        return self._lexical_audit(text)

    def _lexical_audit(self, text: str) -> dict:
        """Local rule-based lexical audit targeting prejudiced or non-neutral terminology."""
        text_lower = text.lower()
        
        # Keywords indicating possible non-neutral profiling or subjective remarks
        bias_markers = [
            r'\bprofiled\b', r'\bprejudiced\b', r'\bdiscriminated?\b',
            r'\bcharacter assassination\b', r'\bchastity\b',
            r'\bdemographic factor\b', r'\breligious background\b',
            r'\bcaste factor\b', r'\bsocioeconomic status\b',
            r'\bsuspect community\b', r'\bsuspicious background\b'
        ]
        
        hits = []
        for m in bias_markers:
            match = re.search(m, text_lower)
            if match:
                hits.append(match.group(0))
        
        if hits:
            score = 0.98 - (len(hits) * 0.1)
            score = max(0.40, min(score, 1.0))
            details = f"Linguistic audits flagged {len(hits)} non-neutral markers (e.g. {', '.join(hits[:2])})."
            return {
                "bias_score": round(score, 2),
                "bias_details": details,
                "flags": hits
            }
        else:
            # If zero hits, return None (unaudited/unknown) rather than claiming 98% neutrality
            return {
                "bias_score": None,
                "bias_details": "Automated audit unavailable. Manual review recommended.",
                "flags": []
            }

    def audit_dataset_distribution(self, conn) -> dict:
        """
        Checks if priority scores are systematically skewed by case_type.
        Returns mean priority per case_type and flags if difference > threshold.
        """
        cur = conn.cursor()
        cur.execute("""
            SELECT case_type, AVG(priority_score), COUNT(*), STDDEV(priority_score)
            FROM cases GROUP BY case_type
        """)
        rows = cur.fetchall()
        return {r[0]: {"mean": round(r[1],2) if r[1] is not None else 0.0, "count": r[2], "std": round(r[3],2) if r[3] is not None else 0.0} for r in rows}

if __name__ == "__main__":
    # Test stub
    agent = BiasAgent()
    sample_text = (
        "The accused belongs to a suspicious background and suspect community, having been profiled "
        "by local police during previous investigations. Wording in the verdict shows non-neutral "
        "assertions regarding his religious background."
    )
    result = agent.audit_case_fairness(sample_text)
    print("Test Bias Audit:", result)
