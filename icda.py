from typing import List

from numpy import extract

from finding_extractor import FindingExtractor
from diagnosis_classifier import DiagnosisClassifier
from term_suggester import TermSuggester
from emr_preprocessor import EMRPreprocessor
from state_tracker import PatientStateTracker

class ICDA(object):
    # TODO: change __init__() to __init__(args) to initialize each component
    def __init__(
            self,
            system_mode: str,
            extract_mode: str,
            finding_extractor: FindingExtractor,
            diagnosis_classifier: DiagnosisClassifier,
            term_suggester: TermSuggester,
            emr_preprocessor: EMRPreprocessor,
            state_tracker: PatientStateTracker = None
        ):
        if system_mode not in ["train", "test", "deploy"]:
            raise ValueError("mode should be 'train', 'test', or 'deploy'")

        self.system_mode = system_mode
        self.extract_mode = extract_mode
        self.finding_extractor = finding_extractor
        self.diagnosis_classifier = diagnosis_classifier
        self.term_suggester = term_suggester
        self.emr_preprocessor = emr_preprocessor
        if self.system_mode != "deploy":
            self.state_tracker = state_tracker

    def generate_support(self, emrs: List[str], n_dx: int = 5) -> List[dict]:
        # preprocess EMRs
        span2pol_l = self.finding_extractor.recognizer.extract_labeled_spans(emrs)
        spans_l, pols_l = self.finding_extractor.recognizer.extract_spans_and_pols(span2pol_l)
        terms_l = self.finding_extractor.normalizer.normalize_term_spans(emrs, spans_l, mode=self.extract_mode)
        text_l = self.emr_preprocessor.add_label_before_every_term(terms_l, labels_l=pols_l)

        # predict top-k diagnoses
        all_logits = self.diagnosis_classifier.predict(text_l)
        dxs_l, probs_l = self.diagnosis_classifier.get_top_dxs_with_probs(all_logits, top_k=n_dx) # get all sorted diagnoses

        # suggest terms
        sug_terms_l = self.term_suggester.suggest_terms_l(text_l, obs_terms_l=terms_l, dxs_l=dxs_l, probs_l=probs_l)
        
        # compile supports
        polname2spans_l = self.finding_extractor.recognizer.get_polname2spans_l(span2pol_l)
        polname2terms_l = self.finding_extractor.recognizer.get_polname2terms_l(terms_l, pols_l)
        dx2_ttype2terms_l = self.term_suggester.classify_sug_terms(sug_terms_l)
        assert len(emrs) == len(polname2spans_l) == len(dxs_l) == len(probs_l) == len(sug_terms_l)
        supports = [
            {
                "emr_display": {
                    "text": emr,
                    "extracted_terms_offsets": polname2spans,
                    "extracted_terms": polname2terms,
                },
                "diagnoses": [
                    {"icd": dx, "name": self.diagnosis_classifier.dx2name[dx], "probability": prob}
                    for dx, prob in zip(dxs, probs)
                ],
                "suggested_terms": {
                    dx: {
                        "symptoms": ttype2terms.get("symptoms", []),
                        "diseases": ttype2terms.get("diseases", [])
                    }
                    for dx, ttype2terms in dx2_ttype2terms.items()
                }
            }
            for emr, polname2spans, polname2terms, dxs, probs, dx2_ttype2terms in zip(emrs, polname2spans_l, polname2terms_l, dxs_l, probs_l, dx2_ttype2terms_l)
        ]
        # TODO: if capstone -> change schema

        return supports