from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders

bert_tokenizer = Tokenizer(WordPiece())
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
bert_tokenizer.pre_tokenizer = Whitespace()
bert_tokenizer.post_tokenizer = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2)
    ]
)

def pretrain_wikitext():
    files = [f'data/wikitext-103-raw/wiki.{split}.raw' for split in ["test", "train", "valid"]]
    trainer = WordPieceTrainer(
    vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    bert_tokenizer.train(trainer, files)
    model_files = bert_tokenizer.model.save("data", "bert-wiki")
    bert_tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")
    bert_tokenizer.save("data/bert-wiki.json")
    
    output = bert_tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.ids)
    result = bert_tokenizer.decode(output.ids)
    print(result)
    
    output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.")
    print(output.tokens)
    
    result = bert_tokenizer.decode(output.ids)
    print(result)
    
    bert_tokenizer.decoder = decoders.WordPiece()
    result = bert_tokenizer.decode(output.ids)
    print(result)

if __name__ == '__main__':
    tokenizer = bert_tokenizer.from_file('data/bert-wiki.json')
    tokens = tokenizer.encode('ciao, come va?')
    print(tokens.ids)

    tokenizer.decoder = decoders.WordPiece()
    print(tokenizer.decode(tokens.ids))
