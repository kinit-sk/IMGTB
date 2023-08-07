from methods.abstract_methods.experiment import Experiment
import numpy as np
import transformers
import re
import torch
import torch.nn.functional as F
import random
import time
from tqdm import tqdm
from methods.utils import get_clf_results

FILL_DICTIONARY = set()

class PertubationBasedExperiment(Experiment):
     def __init__(self, data, model, tokenizer, args, **kwargs): # Add extra parameters if needed
        name = self.__class__.__name__ # Set your own name or leave it set to the class name
        super().__init__(data, name)
        self.base_model = model
        self.base_tokenizer = tokenizer
        self.args = args
     
     def run(self):
        mask_filling_model_name = self.args.mask_filling_model_name
        cache_dir = self.args.cache_dir

        # get mask filling model (for DetectGPT only)
        if self.args.random_fills:
            FILL_DICTIONARY = set()
            for texts in self.data['train']['text'] + self.data['test']['text']:
                for text in texts:
                    FILL_DICTIONARY.update(text.split())
            FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

        int8_kwargs = {}
        half_kwargs = {}
        if self.args.int8:
            int8_kwargs = dict(load_in_8bit=True,
                            device_map='auto', torch_dtype=torch.bfloat16)
        elif self.args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)

        if not self.args.random_fills:
            try:
                n_positions = mask_model.config.n_positions
            except AttributeError:
                n_positions = 512
        else:
            n_positions = 512

        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)

        # perturbation_mode = 'd'
        perturbation_mode = 'z'
        n_perturbations = 10
        t1 = time.time()

        perturbation_results = self.get_perturbation_results(
            self.args, self.data, mask_model, mask_tokenizer, self.base_model, self.base_tokenizer, self.args.span_length, n_perturbations)

        res = self.evaluate_perturbation_results(self.args, perturbation_results, perturbation_mode,
                                        span_length=self.args.span_length, n_perturbations=n_perturbations)
        print(f'{self.name} took %.4f sec' % (time.time() - t1))
        return res
    
     def get_perturbation_results(self, args, data, mask_model, mask_tokenizer, base_model, base_tokenizer, span_length=10, n_perturbations=1):
        load_mask_model(args, mask_model)

        torch.manual_seed(0)
        np.random.seed(0)

        train_text = data['train']['text']
        train_label = data['train']['label']
        test_text = data['test']['text']
        test_label = data['test']['label']

        p_train_text = perturb_texts(args, [x for x in train_text for _ in range(
            n_perturbations)], mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False)
        p_test_text = perturb_texts(args, [x for x in test_text for _ in range(
            n_perturbations)], mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False)

        for _ in range(args.n_perturbation_rounds - 1):
            try:
                p_train_text, p_test_text = perturb_texts(args, p_train_text, mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False), perturb_texts(
                    args, p_test_text, mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False)
            except AssertionError:
                break

        assert len(p_train_text) == len(train_text) * \
            n_perturbations, f"Expected {len(train_text) * n_perturbations} perturbed samples, got {len(p_train_text)}"
        assert len(p_test_text) == len(test_text) * \
            n_perturbations, f"Expected {len(test_text) * n_perturbations} perturbed samples, got {len(p_test_text)}"

        train = []
        test = []
        for idx in range(len(train_text)):
            train.append({
                "text": train_text[idx],
                "label": train_label[idx],
                "perturbed_text": p_train_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            })
        for idx in range(len(test_text)):
            test.append({
                "text": test_text[idx],
                "label": test_label[idx],
                "perturbed_text": p_test_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            })

        # base_model = base_model.to(args.DEVICE)

        return self.compute_perturbation_results(train, test, base_model, base_tokenizer, args)
    
     def compute_perturbation_results(self, train, test, base_model, base_tokenizer, args):
        raise NotImplementedError("Attempted to call an abstract method.")
    
     def evaluate_perturbation_results(self, args, perturbation_results, perturbation_mode,
                                        span_length, n_perturbations):
        raise NotImplementedError("Attempted to call an abstract method.")

        

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def load_mask_model(args, mask_model):
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    # base_model.cpu()
    if not args.random_fills:
        mask_model.to(args.DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, buffer_size, pct, ceil_pct=False):
    tokens = text.split(' ')
    if len(tokens) > 512:
        tokens = tokens[:512]
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, mask_model, mask_tokenizer, mask_top_p, DEVICE):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt",
                            padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True,
                                  top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(args, texts, mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False):
    span_length = args.span_length
    buffer_size = args.buffer_size
    mask_top_p = args.mask_top_p
    pct = args.pct_words_masked
    DEVICE = args.DEVICE
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(
            x, span_length, buffer_size, pct, ceil_pct) for x in texts]
        raw_fills = replace_masks(
            masked_texts, mask_model, mask_tokenizer, mask_top_p, DEVICE)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(idxs)
            for idx, x in enumerate(perturbed_texts):
                if idx in idxs:
                    print(x)
            for idx, x in enumerate(texts):
                if idx in idxs:
                    print(x)
            print(
                f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(
                x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(
                masked_texts, mask_model, mask_tokenizer, mask_top_p, DEVICE)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(
                masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(
                texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * \
                (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(
                tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(
                0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(
                    0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(
                tokens.input_ids, skip_special_tokens=True)
        else:
            print("This here:", FILL_DICTIONARY)
            masked_texts = [tokenize_and_mask(
                x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(
                        f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[
                    0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


def perturb_texts(args, texts, mask_model, mask_tokenizer, base_tokenizer, ceil_pct=False):

    outputs = []
    for i in tqdm(range(0, len(texts), args.chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(args,
                                      texts[i:i + args.chunk_size], mask_model, mask_tokenizer, base_tokenizer, ceil_pct=ceil_pct))
    return outputs