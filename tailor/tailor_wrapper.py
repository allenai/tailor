import imp
import string
from spacy.tokens import Doc
from tailor.steps.process_with_spacy import GetSpacyModel, ProcessWithSpacy
from tailor.steps.get_srl_tags import GetSRLTags
from tailor.steps.random_prompts import _load_default_common_keywords
from tailor.steps.generate_from_prompts import GenerateFromPrompts

from tailor.common.utils import get_srl_tagger
from tailor.common.abstractions import ProcessedSentence, PromptObject
from tailor.common.utils.model_utils import load_generator
from tailor.common.filters.perplex_filter import load_perplex_scorer

from typing import Iterable, Tuple, List
from tailor.common.perturb_function import PerturbStringFunction
from tailor.common.utils.detect_perturbations import detect_perturbations, identify_tags_for_span
from tailor.common.utils.head_prompt_utils import get_unique_tags
from tailor.steps.perturb_prompt import PerturbPromptWithString


class Tailor(object):
    def __init__(self):
        self.spacy_model = None
        self.srl_tagger = None
        self.tailor_generator = None
        self.common_keywords_by_tag = None
        self.perplex_scorer = None

    def _get_tailor_generator(self):
        if self.tailor_generator is None:
            self.tailor_generator = load_generator()
        return self.tailor_generator

    def _get_common_keywords(self):
        if self.common_keywords_by_tag is None:
            self.common_keywords_by_tag = _load_default_common_keywords()
        return self.common_keywords_by_tag

    def _get_perplex_scorer(self):
        if self.perplex_scorer is None:
            self.perplex_scorer = load_perplex_scorer()
        return self.perplex_scorer

    def _get_spacy_model(self,
                         parse=True,
                         spacy_model_name="en_core_web_sm",
                         use_white_space_tokenizer=False):
        if self.spacy_model is None:
            self.spacy_model = GetSpacyModel().run(
                parse=parse,
                spacy_model_name=spacy_model_name,
                use_white_space_tokenizer=use_white_space_tokenizer)
        return self.spacy_model

    def _get_srl_tagger(self):
        if self.srl_tagger is None:
            self.srl_tagger = get_srl_tagger()
        return self.srl_tagger

    def _process(self,
                 sentence: Tuple[Iterable[str], str]
                 ) -> List[ProcessedSentence]:
        """
        Returns the list of sentences with SRL tags.

        Parameters
        ----------
        sentence : :class:`Iterable[str] | str`
            The list of sentences to process. Can also be a single sentence.

        Returns
        -------
        :class:`List[ProcessedSentence]`
            The list of processed sentences with spacy docs and SRL tags.
        """
        if type(sentence) == str:
            sentences = [sentence]
        else:
            sentences = sentence
        spacy_model = self._get_spacy_model()
        spacy_outputs = ProcessWithSpacy().run(
            sentences=sentences, spacy_model=spacy_model)
        # get the SRL tag
        srl_tagger = self._get_srl_tagger()
        processed_sentences = GetSRLTags().run(
            spacy_outputs=spacy_outputs,
            srl_tagger=srl_tagger)
        if type(sentence) == str:
            return processed_sentences[0]
        return processed_sentences

    def default_perturb_func_list(self):
        """
        Returns the list of sentences with SRL tags.

        Parameters
        ----------
        sentence : :class:`Iterable[str] | str`
            The list of sentences to process. Can also be a single sentence.

        Returns
        -------
        :class:`List[str]`
            A list of pre-implemented perturbation functions. These functions
            can be extracted by `tailor.common.perturb_function.PerturbStringFunction.by_name(name)`.
        """
        return PerturbStringFunction.list_available()

    def _infer_target_span(self,
                           selected_span: str,
                           doc: Doc,
                           char_start_idx: int = 0) -> Tuple[int, int]:
        """
        Helpre function for detecting the indexes of a selected span in a sentence.

        Parameters
        ----------
        selected_span : :class:`str`
            The selected span.
        doc : :class:`spacy.tokens.Doc`
            The sentence in the Spacy doc format.
        char_start_idx : :class:`int`
            When to start search for the selected span.

        Returns
        -------
        :class:`Tuple[int, int]`
            The start and end index of the selected span
        """

        spacy_model = self._get_spacy_model()
        tokenizer = spacy_model.tokenizer
        if selected_span is None:
            return [0, len(doc)]
        selected_span = selected_span.strip(string.punctuation)
        selected_span = selected_span.strip()

        if selected_span.lower() not in doc.text.lower():
            print(f"The span does not exist: [{selected_span}] \n\tin [{doc.text}]")
            return [0, len(doc)]
        if len(selected_span) == 0:
            pre_span = doc.text[:char_start_idx].strip()
        else:
            pre_span = doc.text[char_start_idx:].lower().split(
                selected_span.lower())[0]
            pre_span = doc.text[:char_start_idx + len(pre_span)].strip()
        start = len(tokenizer(pre_span)) if pre_span else 0
        length = len(tokenizer(selected_span)) if selected_span else 0
        return [start, start + length]

    def perturb_with_context(self,
                             sentence: str,
                             selected_span: str,
                             to_content: str = None,
                             to_semantic_role: str = None,
                             to_tense: str = None,
                             perplex_thred=50,
                             num_perturbs=10,
                             verbalize=False):
        """
        A special version of `perturb`; returns a list of perturbations but with
        some controls.

        Parameters
        ----------
        sentence : :class:`str`
            A single sentence to perturb.
        selected_span : :class:`str`
            Opional; Denote the specific part to change using a subspan.
        to_content : :class:`str`
            Optional; Keywords that should occur in the generation.
        to_semantic_role : :class:`str`
            Optional; Randomly select some phrase in the current sub-span as keyword, but change the generated semantic role. Accepted list includes
                ['PURPOSE', 'AGENT', 'DISCOURSE', 'MODAL', 'PREDICATE', 'ATTRIBUTE', 
                'PATIENT', 'GOAL', 'END', 'ARG2', 'DIRECTIONAL', 'CAUSE', 'EXTENT', 
                'COMITATIVE', 'TEMPORAL', 'MANNER', 'NEGATION', 'ADVERBIAL', 
                'LOCATIVE', 'VERB']
        to_tense : :class:`str`
            Optional; Randomly select some phrase in the current sub-span as keyword, but change the generated tense. Accepted list includes
        compute_perplexity : :class:`bool`
            Optional; whether to compute the perplexity of the perturbations. Used for filtering out degenerations
        perplex_thred : :class:`int`
            Optional; threshold for filtering out degenerations.
        num_perturbs : :class:`int`
            Optional; number of perturbations to return.
        verbalize : :class:`bool`
            Optional; if `True`, will print a preview of perturbation strategies.

        Returns
        -------
        :class:`List[str]`
            A list of perturbed sentences.
        """
        candidate_inputs = self.detect_possible_perturbs(
            sentence, selected_span, verbalize=False,
            to_content=to_content,
            to_tense=to_tense,
            to_semantic_role=to_semantic_role)
        if to_semantic_role is not None:
            allowed_perturbs = ["change_role"]
        elif to_tense is not None:
            allowed_perturbs = ["change_tense"]
        elif to_content is not None:
            allowed_perturbs = ["change_content"]
        return self.perturb(
            sentence, selected_span,
            candidate_inputs=candidate_inputs,
            allowed_perturbs=allowed_perturbs,
            perplex_thred=perplex_thred,
            num_perturbs=num_perturbs,
            verbalize=verbalize)

    def perturb(self,
                sentence: str,
                selected_span: str = None,
                allowed_perturbs: List[str] = None,
                candidate_inputs: List[PromptObject] = None,
                perplex_thred=50,
                num_perturbs=10, verbalize=False):
        """
        Returns a list of perturbations of a given sentence.

        Parameters
        ----------
        sentence : :class:`str`
            A single sentence to perturb.
        selected_span : :class:`str`
            Opional; Denote the specific part to change using a subspan.
        allowed_perturbs : :class:`List[str]`
            Optional; Names of perturbation types that are allowed.
        candidate_inputs : :class:`List[PromptObject]`
            Optional; candidate input formats.
        compute_perplexity : :class:`bool`
            Optional; whether to compute the perplexity of the perturbations. Used for filtering out degenerations
        perplex_thred : :class:`int`
            Optional; threshold for filtering out degenerations.
        num_perturbs : :class:`int`
            Optional; number of perturbations to return.
        verbalize : :class:`bool`
            Optional; if `True`, will print a preview of perturbation strategies.

        Returns
        -------
        :class:`List[str]`
            A list of perturbed sentences.
        """
        generations = []
        processed_sentence = self._process(sentence)
        if candidate_inputs is None:
            candidate_inputs = self.detect_possible_perturbs(sentence, selected_span, verbalize=False)
        if allowed_perturbs is not None:
            # filter by neededperturbation
            candidate_inputs = [
                prompt for prompt in candidate_inputs
                if any([pname in prompt.description for pname in allowed_perturbs])
            ]
        if verbalize:
            self._print_functions([sentence], [candidate_inputs])
        if len(candidate_inputs) != 0:
            # get models
            if perplex_thred is not None:
                perplex_scorer = self._get_perplex_scorer()
            else:
                perplex_scorer = None
            tailor_generator = self._get_tailor_generator()
            # filter
            generations = GenerateFromPrompts().run(
                processed_sentences=[processed_sentence],
                prompts=[candidate_inputs],
                spacy_model=self.spacy_model,
                compute_perplexity=perplex_thred is not None,
                perplex_scorer=(perplex_scorer.model, perplex_scorer.tokenizer),
                generator=tailor_generator,
            )[0]
        generations = sorted(generations, key=lambda x: (
            x.perplexities.pr_sent, x.perplexities.pr_phrase))
        generations = [g for g in generations if perplex_thred is None or (
            g.perplexities.pr_sent < perplex_thred and g.perplexities.pr_phrase < perplex_thred)]
        generations = generations[:num_perturbs]
        return [g.sentence for g in generations]

    def detect_possible_perturbs(
            self,
            sentence: str,
            selected_span: str = None,
            to_content: str = None,
            to_semantic_role: str = None,
            to_tense: str = None,
            verbalize=True) -> List[PromptObject]:
        """
        Detects possible perturbations of a given sentence, by creating various possible perturbation inputs to Tailor

        Parameters
        ----------
        sentence : :class:`str`
            A single sentence to perturb.
        selected_span : :class:`str`
            Opional; Denote the specific part to change using a subspan.
        to_content : :class:`str`
            Optional; Keywords that should occur in the generation.
        to_semantic_role : :class:`str`
            Optional; Randomly select some phrase in the current sub-span as keyword, but change the generated semantic role. Accepted list includes
                ['PURPOSE', 'AGENT', 'DISCOURSE', 'MODAL', 'PREDICATE', 'ATTRIBUTE', 
                'PATIENT', 'GOAL', 'END', 'ARG2', 'DIRECTIONAL', 'CAUSE', 'EXTENT', 
                'COMITATIVE', 'TEMPORAL', 'MANNER', 'NEGATION', 'ADVERBIAL', 
                'LOCATIVE', 'VERB']
        to_tense : :class:`str`
            Optional; Randomly select some phrase in the current sub-span as keyword, but change the generated tense. Accepted list includes
        verbalize : :class:`bool`
            Optional; (specific to verbs) change the tense (future, present, past).

        Returns
        -------
        :class:`List[PromptObject]`
            A list of perturbation strategies (inputs to Tailor) that can be passed to `tailor.perturb`.
        """
        processed_sentence = self._process(sentence)
        common_keywords_by_tag = self._get_common_keywords()
        start, end = self._infer_target_span(
            selected_span,
            processed_sentence.spacy_doc,
            0)
        candidates = detect_perturbations(
            processed_sentence.spacy_doc,
            start=start,
            end=end,
            predicted=processed_sentence.verbs,
            common_keywords_by_tag=common_keywords_by_tag,
            to_content=to_content,
            to_tense=to_tense,
            to_semantic_role=to_semantic_role
        )
        candidates = [
            PromptObject(
                prompt=prompt_munch.prompt,
                description=prompt_munch.description,
                name=prompt_munch.name)
            for prompt_munch in candidates]
        if verbalize:
            print("DETECTED POSSIBLE CHANGES")
            self._print_functions([sentence], [candidates])
        return candidates

    def _print_functions(self,
                         sentences: Iterable[str],
                         candidate_prompts: Iterable[PromptObject]) -> None:
        assert len(sentences) == len(candidate_prompts)
        for sent, prompt_objects in zip(sentences, candidate_prompts):
            print(f"\nSENTENCE: {sent}")
            if len(prompt_objects) == 0:
                print(f"\t| (No possible perturbations)")
            for prompt_object in prompt_objects:
                print(f"\t| {prompt_object.description}")
                print(f"\t| {prompt_object.prompt}")
        print("\n")
