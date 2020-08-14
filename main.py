from dataset.MarcoPassageDataset import *
from model.BertEmb import *
from util import *
# from model import *

def test(args, dataset, model):
    start = time.time()
    model.eval()
    rets = {}
    with torch.no_grad():
        # get query emb
        q_embs = []
        qid_list = []
        start = time.time()
        batch_len = (len(dataset.queries_dev) + args.batch_size_test - 1) // args.batch_size_test
        for batch in range(batch_len):
            q = dataset.get_dev_q_batch(args, batch)
            qid_list += q['id']
            q_emb = model(q['ids'], q['seg'], q['mask'])
            q_embs.append(q_emb)
            end = time.time()
            print('Build query emb : Batch %d / %d [%s]' % (batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
        print()
        q_embs = torch.cat(q_embs, dim=0)
        # get doc emb
        p_embs = []
        pid_list = []
        start = time.time()
        batch_len = (len(dataset.passages) + args.batch_size_test - 1) // args.batch_size_test
        for batch in range(batch_len):
            p = dataset.get_dev_p_batch(args, batch)
            pid_list += p['id']
            p_emb = model(p['ids'], p['seg'], p['mask'])
            p_embs.append(p_emb)
            end = time.time()
            print('Build doc emb : Batch %d / %d [%s]' % (batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
        print()
        p_embs = torch.cat(p_embs, dim=0)
        # get scores
#        scores = torch.matmul(q_embs, d_embs.transpose(0, 1)).cpu().tolist()
    start = time.time()
    trec_out_fp = open('data/marco_passage/predict.trec', 'w', encoding='utf-8')
    for qid in range(q_emb.size(0)):
        scores = torch.matmul(q_embs[qid:qid+1, :], p_embs.transpose(0, 1)).cpu().view(-1).tolist()
        ranks = sorted(zip(scores, pid_list), reverse=True)[:1000]
        for i in range(len(ranks)):
            trec_out_fp.write('%s Q0 %s %d %f yonsei\n' % (qid_list[qid], ranks[i][1], i + 1, ranks[i][0]))
    trec_out_fp.close()
    trec_eval_res = subprocess.Popen(['data/trec_eval', '-m', 'all_trec', 'data/marco_passage/qrels.dev.tsv', 'data/marco_passage/predict.trec'], stdout=subprocess.PIPE, shell=False)
    out, err = trec_eval_res.communicate()
    lines = out.decode('utf-8').strip().split('\n')
    metrics = {}
    for line in lines[1:]:
        metric, _, value = line.split()
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        metrics[metric.lower()] = value
    return metrics

def train(args, dataset):
    best_recall = 0.0
    model = BertEmb()
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(1, args.total_epoch + 1):
        batch_len = len(dataset.triples_train) // args.batch_size
        loss_cum = []
        start = time.time()
        for batch in range(batch_len):
            q, rd, ud = dataset.get_train_batch(args, batch)
            optimizer.zero_grad()
            q_emb = model(q['ids'], q['seg'], q['mask'])
            rd_emb = model(rd['ids'], rd['seg'], rd['mask'])
            ud_emb = model(ud['ids'], ud['seg'], ud['mask'])
            rel_scores = torch.sum(q_emb * rd_emb, dim=1)
            urel_scores = torch.sum(q_emb * ud_emb, dim=1)
            loss = F.relu(args.margin_const - (rel_scores - urel_scores)).mean()
            loss.backward()
            optimizer.step()
            loss_cum.append(loss.item())
            end = time.time()
            # log
            loss_now = sum(loss_cum) / len(loss_cum)
            print('[E%d] Batch %d / %d  Loss %.6f [%s]' % (epoch, batch + 1, batch_len, loss_now, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
            # eval
            if (batch + 1) % 1000 == 0:
                print()
                metrics = test(args, dataset, model)
                metrics['epoch'] = epoch
                metrics['batch'] = batch
                print('| R@1000 | P@20 | nDCG@20 | MAP |')
                print('| %6.2f | %4.2f | %7.2f | %3.2f |' % (metrics['recall_1000'] * 100, metrics['p_20'] * 100, metrics['ndcg_cut_20'] * 100, metrics['map'] * 100))
                if best_recall < metrics['recall_1000']:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    for name in sorted(os.listdir('save')):
                        if float(name[:-3]) < metrics['recall_1000']:
                            os.remove('save/%s' % name)
                    torch.save(state, 'save/%.4f.pt' % metrics['recall_1000'])
                    best_recall = metrics['recall_1000']
                    print('########## SAVED ##########')
                    model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=5)
    parser.add_argument('--batch_size_test', required=False, type=int, default=50)
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
    # fix random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # create save dir
    if not os.path.exists('save'):
        os.makedirs('save')
    dataset = MarcoPassageDataset()
    q, r, u = dataset.get_train_batch(args, 0)
    # train
    train(args, dataset)

