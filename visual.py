from dataset.MarcoPassageDataset import *
from model.BertEmb import *
from util import *

def stop(word):
    return False
    stopwords = ['.', ',', '"', "'", ';', ':', '?', '!']
    if word in stopwords:
        return True
    if len(word) <= 2:
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=5)
    parser.add_argument('--batch_size_test', required=False, type=int, default=1)
    parser.add_argument('--cls', required=False, type=int, default=-1)
    parser.add_argument('--margin_const', required=False, type=float, default=10.0)
    parser.add_argument('--max_plen', required=False, type=int, default=400)
    parser.add_argument('--max_qlen', required=False, type=int, default=20)
    parser.add_argument('--seed', required=False, type=int, default=1234)
    parser.add_argument('--sep', required=False, type=int, default=-1)
    parser.add_argument('--topic_dim', required=False, type=int, default=500)
    parser.add_argument('--total_epoch', required=False, type=int, default=10)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if args.cls == -1:
        args.cls = tokenizer.vocab['[CLS]']
    if args.sep == -1:
        args.sep = tokenizer.vocab['[SEP]']

    dataset = MarcoPassageDataset()
    model = BertEmb()
    model = model.to(device)
    state_dict = torch.load('save/1.0000.pt')['model']
    model.load_state_dict(state_dict)

    q_tok = tokenizer.tokenize(dataset.queries_dev['58'])[:args.max_qlen]
    q_ids = [args.cls] + tokenizer.convert_tokens_to_ids(q_tok) + [args.sep]
    q_seg = [0 for _ in range(len(q_ids))]
    q_mask = [1 for _ in range(len(q_ids))]

    p_tok = tokenizer.tokenize(dataset.passages['7571934'])[:args.max_plen]
    p_ids = [args.cls] + tokenizer.convert_tokens_to_ids(p_tok) + [args.sep]
    p_seg = [1 for _ in range(len(p_ids))]
    p_mask = [1 for _ in range(len(p_ids))]

    q_ids = torch.tensor([q_ids], dtype=torch.long).to(device)
    q_seg = torch.tensor([q_seg], dtype=torch.long).to(device)
    q_mask = torch.tensor([q_mask], dtype=torch.long).to(device)

    p_ids = torch.tensor([p_ids], dtype=torch.long).to(device)
    p_seg = torch.tensor([p_seg], dtype=torch.long).to(device)
    p_mask = torch.tensor([p_mask], dtype=torch.long).to(device)

    emb_q, seq_q = model(q_ids, q_seg, q_mask, out_seq=True)
    emb_p, seq_p = model(p_ids, p_seg, p_mask, out_seq=True)

    att = torch.matmul(emb_q, seq_p.permute(0, 2, 1)).view(-1).tolist()
    ranks = sorted([(att[i + 1], p_tok[i]) for i in range(len(p_tok)) if not stop(p_tok[i])], reverse=True)
    print('Q:', ' '.join(q_tok))
    print('P:', ' '.join(p_tok))
    print(ranks[:10])
    with open('temp.txt', 'w', encoding='utf-8') as fp:
        fp.write(' '.join(q_tok) + '\n')
        fp.write(' '.join(p_tok) + '\n')
        fp.write(' '.join(list(map(str, att[1:-1]))) + '\n')

