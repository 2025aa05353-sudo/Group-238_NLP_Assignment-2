"""
================================================================================
File Name   : model_utils.py
Purpose     : Inference engine, candidate question generator, answer span 
              extractor, and quantitative evaluation metrics computation 
              (BLEU-4, METEOR, ROUGE-1, ROUGE-L).
              Enforces strict local checkpoint verification without silent fallback
              and initializes NLTK resources conditionally.
================================================================================
Course      : Natural Language Processing - Assignment 2
Program     : M.Tech. in AIML, BITS Pilani (WILP)
Group No    : Group 238
================================================================================
"""

import os
import re
import sys
from pathlib import Path
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# ==============================================================================
# CONDITIONAL NLTK RESOURCE INITIALIZATION (DOWNLOAD ONLY WHEN MISSING)
# ==============================================================================
nltk_resource_map = {
    'tokenizers/punkt': 'punkt',
    'tokenizers/punkt_tab': 'punkt_tab',
    'corpora/wordnet': 'wordnet',
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
    'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng'
}

for resource_path, download_name in nltk_resource_map.items():
    try:
        nltk.find(resource_path)
    except LookupError:
        try:
            nltk.download(download_name, quiet=True)
        except Exception as e:
            print(f"[NLTK Warning] Could not download '{download_name}': {str(e)}", file=sys.stderr)


GENERIC_STOPWORDS = {
    "countries", "country", "world", "majority", "part", "system", "way", 
    "time", "things", "people", "years", "powers", "alliances", "protocols", "it",
    "origin", "history", "growth", "plant", "uses", "nutrition", "vegetable",
    "mountains", "mountain", "nutrients", "nutrient", "food", "crop", "family",
    "groups", "source", "first", "million", "millions", "group", "the", "a", "an",
    "thousands", "hundreds", "century", "centuries", "option", "options"
}


def sanitize_input_text(raw_text: str) -> str:
    """
    Cleans markdown formatting, parenthetical brackets, run-on headers, and line breaks.
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', raw_text)
    text = re.sub(r'[\*\-\•\#]', ' ', text)
    text = re.sub(r'[\(\)]', ' ', text)
    text = re.sub(r'\r?\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z0-9])\.([A-Z])', r'\1. \2', text)
    return text.strip()


# ==============================================================================
# QUESTION GENERATOR ENGINE
# ==============================================================================
class QuestionGenerator:
    def __init__(self, model_name: str = "./saved_model"):
        """
        Initializes tokenizer and model dynamically based on requested model_name.
        Raises FileNotFoundError if a requested local model directory does not exist or is empty.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.requested_model_name = model_name
        
        target_path = Path(model_name)
        
        # Strict validation for local paths: No silent fallback to online models
        if str(model_name).startswith(".") or target_path.is_absolute():
            if target_path.exists() and target_path.is_dir() and len(os.listdir(target_path)) > 0:
                model_identifier = str(target_path)
            else:
                raise FileNotFoundError(
                    f"Requested local model directory '{model_name}' was not found or is empty. "
                    "Please download and extract the model weights package using the GitHub Downloader."
                )
        else:
            model_identifier = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, context: str, answer: str = "", num_beams: int = 2, num_return_sequences: int = 1):
        """
        Generates candidate question(s) given context passage and optional target answer.
        Returns a list of candidate strings when num_return_sequences > 1, or a single string otherwise.
        """
        try:
            clean_context = sanitize_input_text(context)
            clean_answer = answer.strip()

            if clean_answer and clean_answer in clean_context:
                highlighted = clean_context.replace(clean_answer, f"<hl> {clean_answer} <hl>", 1)
                input_text = f"generate question: context: {highlighted}"
            elif clean_answer:
                input_text = f"answer: {clean_answer} context: {clean_context}"
            else:
                input_text = f"generate question: context: {clean_context}"

            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)

            effective_returns = min(num_return_sequences, num_beams) if num_beams > 1 else 1

            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=num_beams,
                num_return_sequences=effective_returns,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

            decoded_questions = [
                re.sub(r'[\*\_]', '', self.tokenizer.decode(out, skip_special_tokens=True)).strip()
                for out in outputs
            ]

            if num_return_sequences > 1:
                return decoded_questions
            return decoded_questions[0] if decoded_questions else "Could not generate question."
        except Exception as e:
            err_msg = f"Generation Error: {str(e)}"
            return [err_msg] if num_return_sequences > 1 else err_msg

    def extract_spans_and_generate(self, context: str, limit: int = 5, num_beams: int = 2):
        cleaned_full_text = sanitize_input_text(context)
        sentences = nltk.sent_tokenize(cleaned_full_text)
        results = []
        seen_answers = set()

        for sent in sentences:
            if len(results) >= limit:
                break

            candidate_spans = []

            compound_matches = re.findall(
                r'\b(?:the\s+)?(?:[A-Z][a-zA-Z0-9_]*(?:\s+(?:and|or|of)\s+(?:the\s+)?[A-Z][a-zA-Z0-9_]*)+)\b', 
                sent
            )
            candidate_spans.extend(compound_matches)

            proper_noun_matches = re.findall(
                r'\b(?:[A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*|\s+I{1,3}|\s+IV|\s+V|\s+VI|\s+VII|\s+VIII|\s+IX|\s+X)*)\b', 
                sent
            )
            candidate_spans.extend(proper_noun_matches)

            date_matches = re.findall(
                r'\b(?:\d{1,2}(?:st|nd|rd|th)\s+century|\d{4}[–\-]\d{4}|\d{4}(?:\s+to\s+\d{4})?)\b', 
                sent
            )
            candidate_spans.extend(date_matches)

            tokens = nltk.word_tokenize(sent)
            try:
                tagged = nltk.pos_tag(tokens)
                current_chunk = []
                for word, tag in tagged:
                    if tag in ("NNP", "NNPS") or (tag in ("NN", "NNS") and word.lower() not in GENERIC_STOPWORDS):
                        current_chunk.append(word)
                    else:
                        if len(current_chunk) > 1:
                            candidate_spans.append(" ".join(current_chunk))
                        current_chunk = []
                if len(current_chunk) > 1:
                    candidate_spans.append(" ".join(current_chunk))
            except Exception:
                pass

            valid_candidates = []
            for span in candidate_spans:
                span_clean = span.strip()
                if span_clean.lower().startswith("the ") and not (" and " in span_clean.lower() or " or " in span_clean.lower()):
                    span_clean = span_clean[4:].strip()

                if (
                    span_clean 
                    and len(span_clean) > 2 
                    and span_clean.lower() not in GENERIC_STOPWORDS 
                    and span_clean in sent
                ):
                    valid_candidates.append(span_clean)

            valid_candidates = sorted(list(set(valid_candidates)), key=len, reverse=True)

            for span in valid_candidates:
                span_lower = span.lower()
                if span_lower not in seen_answers:
                    seen_answers.add(span_lower)
                    
                    generated_q = self.generate(context=cleaned_full_text, answer=span, num_beams=num_beams, num_return_sequences=1)
                    results.append((sent, span, generated_q))
                    
                    if len(results) >= limit:
                        return results
                    break

        return results


# ==============================================================================
# COMPREHENSIVE METRICS EVALUATION SUITE (BLEU, METEOR, ROUGE)
# ==============================================================================
def compute_metrics(reference_question: str, candidate_question: str) -> dict:
    try:
        ref_tokens_list = [nltk.word_tokenize(reference_question.lower())]
        cand_tokens = nltk.word_tokenize(candidate_question.lower())

        # 1. BLEU-4 Score
        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(ref_tokens_list, cand_tokens, smoothing_function=smooth)

        # 2. METEOR Score
        meteor_val = meteor_score(ref_tokens_list, cand_tokens)

        # 3. ROUGE-1 & ROUGE-L Scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_results = scorer.score(reference_question, candidate_question)

        return {
            "BLEU-4": round(bleu_score, 4),
            "METEOR": round(meteor_val, 4),
            "ROUGE-1 (F1)": round(rouge_results['rouge1'].fmeasure, 4),
            "ROUGE-L (F1)": round(rouge_results['rougeL'].fmeasure, 4)
        }
    except Exception as e:
        return {
            "BLEU-4": 0.0,
            "METEOR": 0.0,
            "ROUGE-1 (F1)": 0.0,
            "ROUGE-L (F1)": 0.0,
            "Error": str(e)
        }