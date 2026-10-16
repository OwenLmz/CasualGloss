from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    for text in [
        "<WAIT>",
        "▁▁",
        "@@",
        " @@",
        "GO @@",
        "GO DOWN @@",
        "<|wait|>",
        "[WAIT]",
        " WAIT",
        "<END>",
    ]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
        print(repr(text), token_ids, token_pieces, decoded_pieces)


if __name__ == "__main__":
    main()
