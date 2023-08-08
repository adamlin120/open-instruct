from transformers import AutoModelForCausalLM, AutoTokenizer

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('local_path', type=str, help='local path to the model')
    parser.add_argument('remote_path', type=str, help='remote path to the model')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.local_path)
    tokenizer.push_to_hub(args.remote_path)
    model = AutoModelForCausalLM.from_pretrained(args.local_path)
    model.push_to_hub(args.remote_path)
