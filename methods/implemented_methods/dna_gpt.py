
## MIT License
##
## Copyright (c) 2023 Xianjun (Nolan) Yang
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
##SOFTWARE.
##
## Official release for paper DNA-GPT: https://arxiv.org/abs/2305.17359
## Code adopted from https://github.com/Xianjun-Yang/DNA-GPT

from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment

import transformers
import torch
import nltk
import openai
import re
import six
import spacy
import numpy as np

from nltk.stem.porter import PorterStemmer
from rouge_score.rouge_scorer import _create_ngrams
from openai import OpenAI

class DNAGPT(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config)
        self.openai_key = config.get("openai_key", None) ### OpenAI AIP key for access to the re-generation service, users are suggested to delete the key after usage to avoid potential leakage of key
        if self.openai_key is not None:
            openai.api_key = self.openai_key
        self.model_name = config.get("model_name", "")
        self.temperature = config.get("temperature", 0.7)  ### This parameter controls text quality of chatgpt, by default it was set to 0.7 in the website version of ChatGPT.
        self.max_new_tokens = config.get("max_new_tokens", 300) ### maximum length of generated texts from chatgpt
        self.regen_number = config.get("regen_number", 30)  ### for faster response, users can set this value to smaller ones, such as 20 or 10, which will degenerate performance a little bit
        self.question = config.get("question", "")
        self.truncate_ratio = config.get("truncate_ratio", 0.5)
        self.scoring_function = config.get("scoring_function", "blackbox") # options are ["blackbox", "whitebox"]
        self.PorterStemmer = PorterStemmer()
        self.DEVICE = config["DEVICE"]
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name).to(self.DEVICE)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        
        # Remember to specify threshold or threshold_estimation_params (see methods/abstract_methods/metric_based_experiment.py)

        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

        self.stopwords = self.nlp.Defaults.stop_words

        nltk.download('punkt')


    def criterion_fn(self, text: str):
        if self.scoring_function == "blackbox":
            res, _ = self.compute_bscore(text)
            return np.array([res])
        if self.scoring_function == "whitebox":
            return np.array([self.compute_wscore(text)])
        else:
            raise ValueError("Unknown scoring function. Aborting...")

    def tokenize(self, text, stemmer, stopwords=[]):
        """Tokenize input text into a list of tokens.

        This approach aims to replicate the approach taken by Chin-Yew Lin in
        the original ROUGE implementation.

        Args:
        text: A text blob to tokenize.
        stemmer: An optional stemmer.

        Returns:
        A list of string tokens extracted from input text.
        """

        # Convert everything to lowercase.
        text = text.lower()
        # Replace any non-alpha-numeric characters with spaces.
        text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))

        tokens = re.split(r"\s+", text)
        if stemmer:
            # Only stem words more than 3 characters long.
            tokens = [stemmer.stem(x) if len(
                x) > 3 else x for x in tokens if x not in self.stopwords]

        # One final check to drop any empty or invalid tokens.
        tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]

        return tokens


    def get_score_ngrams(self, target_ngrams, prediction_ngrams):
        """calcualte overlap ratio of N-Grams of two documents.

        Args:
        target_ngrams: N-Grams set of targe document.
        prediction_ngrams: N-Grams set of reference document.

        Returns:
        ratio of intersection of N-Grams of two documents, together with dict list of [overlap N-grams: count]
        """
        intersection_ngrams_count = 0
        ngram_dict = {}
        for ngram in six.iterkeys(target_ngrams):
            intersection_ngrams_count += min(target_ngrams[ngram],
                                            prediction_ngrams[ngram])
            ngram_dict[ngram] = min(target_ngrams[ngram], prediction_ngrams[ngram])
        target_ngrams_count = sum(target_ngrams.values()) # prediction_ngrams
        return intersection_ngrams_count / max(target_ngrams_count, 1), ngram_dict


    def get_ngram_info(self, article_tokens, summary_tokens, _ngram):
        """calculate N-Gram overlap score of two documents
        It use _create_ngrams in rouge_score.rouge_scorer to get N-Grams of two docuemnts, then revoke get_score_ngrams method to calucate overlap score
        Args:
        article_tokens: tokens of one document.
        summary_tokens: tokens of another document.

        Returns:
        ratio of intersection of N-Grams of two documents, together with dict list of [overlap N-grams: count], total overlap n-gram count
        """
        article_ngram = _create_ngrams( article_tokens , _ngram)
        summary_ngram = _create_ngrams( summary_tokens , _ngram)
        ngram_score, ngram_dict = self.get_score_ngrams( article_ngram, summary_ngram) 
        return ngram_score, ngram_dict, sum( ngram_dict.values() )


    def N_gram_detector(self, ngram_n_ratio):
        """calculate N-Gram overlap score from N=3 to N=25
        Args:
        ngram_n_ratio: a list of ratio of N-Gram overlap scores, N is from 1 to 25.

        Returns:
        N-Gram overlap score from N=3 to N=25 with decay weighting n*log(n)
        """
        score = 0
        non_zero = []

        for idx, key in enumerate(ngram_n_ratio):
            if idx in range(3) and 'score' in key or 'ratio' in key:
                score += 0. * ngram_n_ratio[key]
                continue
            if 'score' in key or 'ratio' in key:
                score += (idx+1) * np.log((idx+1)) * ngram_n_ratio[key]
                if ngram_n_ratio[key] != 0:
                    non_zero.append(idx+1)
        return score / (sum(non_zero) + 1e-8)


    def N_gram_detector_ngram(self, ngram_n_ratio):
        """sort the dictionary of N-gram key according to their counts
        Args:
        ngram_n_ratio: a list of ratio of N-Gram overlap scores, N is from 1 to 25.

        Returns:
        sorted dictionary of N-gram [key,value] according to their value counts
        """
        ngram = {}
        for idx, key in enumerate(ngram_n_ratio):
            if idx in range(3) and 'score' in key or 'ratio' in key:
                continue
            if 'ngramdict' in key:
                dict_ngram = ngram_n_ratio[key]

                for key_, value_ in dict_ngram.items():
                    ngram[key_] = idx
        sorted_dict = dict(
            sorted(ngram.items(), key=lambda x: x[1], reverse=True))
        return sorted_dict


    def tokenize_nltk(self, text):
        """tokenize text using word tokenizer
        Args:
        text: input text

        Returns:
        tokens of words
        """
        tokens = nltk.word_tokenize(text)
        return tokens


    def get_ngrams(self, tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams


    def getOverlapedTokens(self, text1, text2):
        """get overlap of word tokens of two documents
        Args:
        text1: input text1
        text2: input text2

        Returns:
        dict of [token:count] of overlape words in two texts
        """
        overlap_dict = {}
        tokens1 = self.tokenize_nltk(text1.lower())
        tokens2 = self.tokenize_nltk(text2.lower())
        for n in range(3, 25):
            ngrams1 = self.get_ngrams(tokens1, n)
            ngrams2 = self.get_ngrams(tokens2, n)
            ngrams_set1 = set(ngrams1)
            ngrams_set2 = set(ngrams2)
            overlap = ngrams_set1.intersection(ngrams_set2)
            for element in overlap:
                overlap_dict[element] = n
        return overlap_dict


    def truncate_string_by_words(self, string, max_words):
        words = string.split()
        if len(words) <= max_words:
            return string
        else:
            truncated_words = words[:max_words]
            return ' '.join(truncated_words)


    def generate_openai_completion(self, model_name: str, prompt: str, max_new_tokens: int, temperature: int, n: int) -> str:
        client = OpenAI(api_key=openai.api_key)
        completion = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=n,
            logprobs=1,
        )
        return list(map(lambda choice: choice["message"]["content"]), completion["choices"]), list(map(lambda choice: np.mean([token.logprob for token in choice.logprobs.content]), completion["choices"]))
    

    def generate_hf_completion(self, model_name: str, prompt: str, max_new_tokens: int, temperature: int, n: int) -> str:
        tokens = self.tokenizer(prompt, truncation=True, return_tensors="pt").to(self.DEVICE)
        
        if max_new_tokens == 0:
            labels = tokens.input_ids
            scores = self.model(**tokens, labels=labels).logits
            seq_logprob = 0
            for idx, token_id in enumerate(tokens.input_ids[0]):
                seq_logprob += scores[0][idx][token_id]
            return [prompt], seq_logprob/len(tokens.input_ids[0])
        
        out = self.model.generate(**tokens,
                        return_dict_in_generate=True, 
                        output_scores=True, 
                        max_new_tokens=max_new_tokens, 
                        num_beams=self.regen_number,
                        num_return_sequences=self.regen_number)
        results = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        return results, out.sequences_scores.tolist()


    def generate_completion(self, model_name: str, prompt: str, max_new_tokens: int, temperature: int, n: int) -> str:

        if self.openai_key is not None:
            return self.generate_openai_completion(model_name, prompt, max_new_tokens, temperature, n)
        else:
            return self.generate_hf_completion(model_name, prompt, max_new_tokens, temperature, n)


    def compute_wscore(self, text: str) -> bool:
        max_words = 350 #### can be adjusted
        truncated_text = self.truncate_string_by_words(text, max_words)
        ngram_overlap_count =[]
        input_text = truncated_text
        human_prefix_prompt = input_text[:int(self.truncate_ratio*len(input_text))]
        generated_texts, scores = self.generate_completion(self.model_name, human_prefix_prompt, self.max_new_tokens, self.temperature, self.regen_number)
        _, original_log_prob = self.generate_completion(self.model_name, truncated_text, 0, self.temperature, 1)
        regen_log_prob = np.mean(scores)
        wscore = original_log_prob - regen_log_prob

        return wscore




    def compute_bscore(self, text):
        """detect if give text is generated by chatgpt 
        Args:
        text: input text 
        option: target model to be checked: gpt-3.5-turbo or gpt-4-0314.

        Returns:
        decision: boolean value that indicate if the text input is generated by chatgpt with version as option
        most_matched_generatedtext[0]: most matched re-generated text by chatgpt using half prefix of the input text as the prompt
        """
        max_words = 350 #### can be adjusted
        text = self.truncate_string_by_words(text, max_words)
        ngram_overlap_count =[]
        input_text = text
        human_prefix_prompt = input_text[:int(self.truncate_ratio*len(input_text))]
        generated_texts, _ = self.generate_completion(self.model_name, human_prefix_prompt, self.max_new_tokens, self.temperature, self.regen_number)
        
        input_remaining = input_text[int(self.truncate_ratio*len(input_text)):]
        input_remaining_tokens = self.tokenize(input_remaining, stemmer=self.PorterStemmer)

        temp = []
        mx = 0
        mx_v = 0
        for i in range(self.regen_number):  # len(human_half)
            temp1 = {}
            gen_text = generated_texts[i]

            ###### optional #######
            gen_text_ = self.truncate_string_by_words(gen_text, max_words-150)

            gpt_generate_tokens = self.tokenize(
                gen_text_, stemmer=self.PorterStemmer)
            if len(input_remaining_tokens) == 0 or len(gpt_generate_tokens) == 0:
                continue

            for _ngram in range(1, 25):
                ngram_score, ngram_dict, overlap_count = self.get_ngram_info(
                    input_remaining_tokens, gpt_generate_tokens, _ngram)
                temp1['human_truncate_ngram_{}_score'.format(
                    _ngram)] = ngram_score / len(gpt_generate_tokens)
                temp1['human_truncate_ngram_{}_ngramdict'.format(
                    _ngram)] = ngram_dict
                temp1['human_truncate_ngram_{}_count'.format(
                    _ngram)] = overlap_count

                if overlap_count > 0:
                    if _ngram > mx_v:
                        mx_v = _ngram
                        mx = i

            temp.append({'machine': temp1})

        ngram_overlap_count.append(temp)
        gpt_scores = []

        top_overlap_ngram = []
        max_ind_list = []
        most_matched_generatedtext = []
        for instance in ngram_overlap_count:
            human_score = []
            gpt_score = []

            for i in range(len(instance)):
                # human_score.append(N_gram_detector(instance[i]['human']))
                gpt_score.append(self.N_gram_detector(instance[i]['machine']))
                top_overlap_ngram.append(
                    self.N_gram_detector_ngram(instance[i]['machine']))

            # human_scores.append(sum(human_score))
            gpt_scores.append(np.mean(gpt_score))
            max_value = max(gpt_score)
            # print(len(gpt_score))
            max_index = mx  # gpt_score.index(max_value)
            max_ind_list.append(max_index)
            most_matched_generatedtext.append(
                generated_texts[max_index])
            
        return gpt_scores[0], most_matched_generatedtext[0]