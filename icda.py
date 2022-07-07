from typing import List

class ICDA(object):
    def __init__(
            self,
            mode: str,
            finding_extractor: FindingExtractor,
            diagnosis_classifier: DiagnosisClassifier,
            term_suggester: TermSuggester,
            emr_preprocessor: EMRPreprocessor,
            state_tracker: PatientStateTracker
        ):
        if mode not in ["train", "test", "deploy"]:
            raise ValueError("mode should be 'train', 'test', or 'deploy'")

        self.mode = mode
        self.finding_extractor = finding_extractor
        self.diagnosis_classifier = diagnosis_classifier
        self.term_suggester = term_suggester
        self.emr_preprocessor = emr_preprocessor
        self.state_tracker = state_tracker

    def predict(self, emrs: List[str]) -> List[dict]:
        # preprocess EMRs
        emrs = self.emr_preprocessor.preprocess(emrs)

        # predict top-k diagnoses
        logits = self.diagnosis_classifier.predict(emrs)
        