from util import *

class MarcoPassageDataset:
    def __init__(self):
        # build tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        used_pid = set()

        # load queries (train)
        self.queries_train = {}
        start = time.time()
        fp = open('data/marco_passage/queries.train.tsv', 'r', encoding='utf-8')
        for line in fp:
            qid, query = line.strip().split('\t')
            self.queries_train[qid] = query
            end = time.time()
            print('Load Queries (Train) : %d queries [%s]' % (len(self.queries_train), format_time(start, end, 808731, len(self.queries_train))), end='\r', flush=True)
        fp.close()
        print()

        # load triples (train)
        self.triples_train = []
        start = time.time()
        fp = open('data/marco_passage/qidpidtriples.train.full.tsv', 'r', encoding='utf-8')
        for line in fp:
            triple = line.strip().split('\t')
            self.triples_train.append(triple)
            used_pid.add(triple[1])
            used_pid.add(triple[2])
            end = time.time()
            print('Load Triples (Train) : %d triples [%s]' % (len(self.triples_train), format_time(start, end, 100000, len(self.triples_train))), end='\r', flush=True)
            if len(self.triples_train) >= 100000:
                break
        fp.close()
        print()

        # load queries (dev)
        self.queries_dev = {}
        start = time.time()
        fp = open('data/marco_passage/queries.dev.tsv', 'r', encoding='utf-8')
        for line in fp:
            qid, query = line.strip().split('\t')
            self.queries_dev[qid] = query
            end = time.time()
            print('Load Quries (Dev) : %d queries [%s]' % (len(self.queries_dev), format_time(start, end, 1000, len(self.queries_dev))), end='\r', flush=True) # 101093
            if len(self.queries_dev) >= 1000:
                break
        fp.close()
        print()

        # load qrels (dev)
        self.qrels_dev = {}
        start = time.time()
        fp = open('data/marco_passage/qrels.dev.tsv', 'r', encoding='utf-8')
        for line in fp:
            qid, _, pid, _ = line.strip().split('\t')
            if not qid in self.qrels_dev:
                self.qrels_dev[qid] = []
            self.qrels_dev[qid].append(pid)
            used_pid.add(pid)
            end = time.time()
            print('Load Qrels (Dev) : %d qrels [%s]' % (len(self.qrels_dev), format_time(start, end, 55578, len(self.qrels_dev))), end='\r', flush=True)
        fp.close()
        print()

        # load passages
        self.passages = {}
        start = time.time()
        fp = open('data/marco_passage/collection.tsv', 'r', encoding='utf-8')
        for line in fp:
            pid, passage = line.strip().split('\t')
            if not pid in used_pid:
                continue
            self.passages[pid] = passage
            end = time.time()
            print('Load Passages : %d passages [%s]' % (len(self.passages), format_time(start, end, 166310, len(self.passages))), end='\r', flush=True) #8841823 
        fp.close()
        print()

    def get_train_batch(self, args, batch):
        q_ids, q_seg, q_mask = [], [], []
        r_ids, r_seg, r_mask = [], [], []
        u_ids, u_seg, u_mask = [], [], []
        CLS = self.tokenizer.vocab['[CLS]']
        SEP = self.tokenizer.vocab['[SEP]']
        for i in range(args.batch_size * batch, args.batch_size * (batch + 1)):
            # query
            qid, rid, uid = self.triples_train[i]
            qq_ids = self.tokenizer.tokenize(self.queries_train[qid])
            qq_ids = self.tokenizer.convert_tokens_to_ids(qq_ids)[:args.max_qlen]
            qq_ids = [CLS] + qq_ids + [SEP]
            qq_seg = [0 for _ in range(len(qq_ids))]
            qq_mask = [1 for _ in range(len(qq_ids))]
            while len(qq_ids) < args.max_qlen + 2:
                qq_ids.append(0)
                qq_seg.append(0)
                qq_mask.append(0)
            q_ids.append(qq_ids)
            q_seg.append(qq_seg)
            q_mask.append(qq_mask)

            # rel passage
            rr_ids = self.tokenizer.tokenize(self.passages[rid])
            rr_ids = self.tokenizer.convert_tokens_to_ids(rr_ids)[:args.max_plen]
            rr_ids = [CLS] + rr_ids + [SEP]
            rr_seg = [1 for _ in range(len(rr_ids))]
            rr_mask = [1 for _ in range(len(rr_ids))]
            while len(rr_ids) < args.max_plen + 2:
                rr_ids.append(0)
                rr_seg.append(0)
                rr_mask.append(0)
            r_ids.append(rr_ids)
            r_seg.append(rr_seg)
            r_mask.append(rr_mask)

            # urel passage
            uu_ids = self.tokenizer.tokenize(self.passages[uid])
            uu_ids = self.tokenizer.convert_tokens_to_ids(uu_ids)[:args.max_plen]
            uu_ids = [CLS] + uu_ids + [SEP]
            uu_seg = [1 for _ in range(len(uu_ids))]
            uu_mask = [1 for _ in range(len(uu_ids))]
            while len(uu_ids) < args.max_plen + 2:
                uu_ids.append(0)
                uu_seg.append(0)
                uu_mask.append(0)
            u_ids.append(uu_ids)
            u_seg.append(uu_seg)
            u_mask.append(uu_mask)

        # build tensor
        q_ids = torch.tensor(q_ids, dtype=torch.long).to(device)
        q_seg = torch.tensor(q_seg, dtype=torch.long).to(device)
        q_mask = torch.tensor(q_mask, dtype=torch.long).to(device)
        r_ids = torch.tensor(r_ids, dtype=torch.long).to(device)
        r_seg = torch.tensor(r_seg, dtype=torch.long).to(device)
        r_mask = torch.tensor(r_mask, dtype=torch.long).to(device)
        u_ids = torch.tensor(u_ids, dtype=torch.long).to(device)
        u_seg = torch.tensor(u_seg, dtype=torch.long).to(device)
        u_mask = torch.tensor(u_mask, dtype=torch.long).to(device)

        q = {
            'ids': q_ids,
            'seg': q_seg,
            'mask': q_mask
        }
        r = {
            'ids': r_ids,
            'seg': r_seg,
            'mask': r_mask
        }
        u = {
            'ids': u_ids,
            'seg': u_seg,
            'mask': u_mask
        }
        return q, r, u

    def get_dev_q_batch(self, args, batch):
        qid_list = sorted(self.queries_dev)
        q_id, q_ids, q_seg, q_mask = [], [], [], []
        CLS = self.tokenizer.vocab['[CLS]']
        SEP = self.tokenizer.vocab['[SEP]']
        for i in range(args.batch_size_test * batch, args.batch_size_test * (batch + 1)):
            if len(qid_list) <= i:
                break
            qq_ids = self.tokenizer.tokenize(self.queries_dev[qid_list[i]])
            qq_ids = self.tokenizer.convert_tokens_to_ids(qq_ids)[:args.max_qlen]
            qq_ids = [CLS] + qq_ids + [SEP]
            qq_seg = [0 for _ in range(len(qq_ids))]
            qq_mask = [1 for _ in range(len(qq_ids))]
            while len(qq_ids) < args.max_qlen + 2:
                qq_ids.append(0)
                qq_seg.append(0)
                qq_mask.append(0)
            q_id.append(qid_list[i])
            q_ids.append(qq_ids)
            q_seg.append(qq_seg)
            q_mask.append(qq_mask)

        # build tensor
        q_ids = torch.tensor(q_ids, dtype=torch.long).to(device)
        q_seg = torch.tensor(q_seg, dtype=torch.long).to(device)
        q_mask = torch.tensor(q_mask, dtype=torch.long).to(device)

        q = {
            'id': q_id,
            'ids': q_ids,
            'seg': q_seg,
            'mask': q_mask
        }
        return q

    def get_dev_p_batch(self, args, batch):
        pid_list = sorted(self.passages)
        p_id, p_ids, p_seg, p_mask = [], [], [], []
        CLS = self.tokenizer.vocab['[CLS]']
        SEP = self.tokenizer.vocab['[SEP]']
        for i in range(args.batch_size_test * batch, args.batch_size_test * (batch + 1)):
            if len(pid_list) <= i:
                break
            pp_ids = self.tokenizer.tokenize(self.passages[pid_list[i]])
            pp_ids = self.tokenizer.convert_tokens_to_ids(pp_ids)
            pp_ids = [CLS] + pp_ids + [SEP]
            pp_seg = [0 for _ in range(len(pp_ids))]
            pp_mask = [1 for _ in range(len(pp_ids))]
            while len(pp_ids) < args.max_plen + 2:
                pp_ids.append(0)
                pp_seg.append(0)
                pp_mask.append(0)
            p_id.append(pid_list[i])
            p_ids.append(pp_ids)
            p_seg.append(pp_seg)
            p_mask.append(pp_mask)

        # build tensor
        p_ids = torch.tensor(p_ids, dtype=torch.long).to(device)
        p_seg = torch.tensor(p_seg, dtype=torch.long).to(device)
        p_mask = torch.tensor(p_mask, dtype=torch.long).to(device)

        p = {
            'id': p_id,
            'ids': p_ids,
            'seg': p_seg,
            'mask': p_mask
        }
        return p

if __name__ == '__main__':
    dataset = MarcoPassageDataset()

