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
            front_end: str,
            finding_extractor: FindingExtractor,
            diagnosis_classifier: DiagnosisClassifier,
            term_suggester: TermSuggester,
            emr_preprocessor: EMRPreprocessor,
            state_tracker: PatientStateTracker = None
        ):
        if system_mode not in ["train", "test", "deploy"]:
            raise ValueError("system_mode should be 'train', 'test', or 'deploy'")
        if front_end not in ["ss_dx", "unified"]:
            raise ValueError("front_end should be 'ss_dx' or 'unified'")

        self.system_mode = system_mode
        self.extract_mode = extract_mode
        self.front_end = front_end
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
        sug_terms_d_l = self.term_suggester.suggest_terms_l(text_l, obs_terms_l=terms_l, dxs_l=dxs_l, probs_l=probs_l)
        
        # compile supports
        polname2spans_l = self.finding_extractor.recognizer.get_polname2spans_l(span2pol_l)
        polname2terms_l = self.finding_extractor.recognizer.get_polname2terms_l(terms_l, pols_l)
        if self.front_end == "ss_dx":
            dx2_ttype2terms_l = self.term_suggester.classify_sug_terms(sug_terms_d_l)
        assert len(emrs) == len(polname2spans_l) == len(dxs_l) == len(probs_l) == len(sug_terms_d_l)
        supports = [
            {
                "emr_display": {
                    "text": emrs[i],
                    "extracted_terms_offsets": polname2spans_l[i],
                    "extracted_terms": polname2terms_l[i],
                },
                "diagnoses": [
                    {"icd": dx, "name": self.diagnosis_classifier.dx2name[dx], "probability": prob}
                    for dx, prob in zip(dxs_l[i], probs_l[i])
                ],
                "suggested_terms": {
                    dx: {
                        "symptoms": ttype2terms.get("symptoms", []),
                        "diseases": ttype2terms.get("diseases", [])
                    }
                    for dx, ttype2terms in dx2_ttype2terms_l[i].items()
                } if (self.front_end == "ss_dx")
                else {
                    dx: sug_terms for dx, sug_terms in sug_terms_d_l[i].items()
                }
            }
            for i in range(len(emrs))
        ]
        # TODO: if capstone -> change schema

        return supports