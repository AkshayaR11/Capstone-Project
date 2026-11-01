from agents.prioritization.scoring import priority_components

class ExplainabilityAgent:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.baseline = {
            "case_type": "civil", "num_ipc_sections": 0, "num_cpc_sections": 0,
            "num_precedents": 0, "total_words": 0, "case_age_days": None,
            "max_severity_score": 3.0, "immediate_threat_flag": 0,
            "societal_impact_score": 1,
        }

    def compute_marginal_attributions(self, case_features: dict) -> dict:
        """
        Return the exact weighted score components for the deterministic policy.
        """
        return self._fallback_attributions(case_features)

    def _fallback_attributions(self, case_features: dict) -> dict:
        """Return the exact weighted components used by compute_priority_score."""
        components = priority_components(
            case_features.get("max_severity_score", 3.0),
            case_features.get("societal_impact_score", 1),
            case_features.get("immediate_threat_flag", 0),
            case_features.get("case_age_days"),
            case_features.get("case_type", "civil"),
        )
        rounded = {key: round(value, 2) for key, value in components.items()}
        displayed_total = round(min(sum(components.values()), 10.0), 2)
        rounded["recency_contribution"] = round(
            rounded["recency_contribution"] + displayed_total - sum(rounded.values()), 2
        )
        return rounded

if __name__ == "__main__":
    # Test stub
    agent = ExplainabilityAgent()
    sample_features = {
        "case_type": "criminal",
        "num_ipc_sections": 3,
        "num_cpc_sections": 0,
        "num_precedents": 8,
        "total_words": 1500,
        "case_age_days": 1825.0,
        "max_severity_score": 8.0,
        "immediate_threat_flag": 1,
        "societal_impact_score": 3
    }
    explanation = agent.compute_marginal_attributions(sample_features)
    print("Test local feature attributions:", explanation)
