import argparse
from argparse import Namespace

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def main(config: Namespace) -> None:
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model.save_pretrained(config.save_path)
    tokenizer.save_pretrained(config.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="HF model quantization using Bitsandbytes")
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    args = parser.parse_args()
    main(args)
