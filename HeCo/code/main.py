import numpy
import torch

# from HeCo.code.utils.evaluate import run_kmeans
from utils.evaluate import run_kmeans
from utils import load_data, set_params, evaluate
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random


warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    #  return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test
    #  author 和 subject的邻居节点、 pas的特征向量、 元路径、 正样本矩阵、 标签、 训练集、 验证集、 测试集
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0
    period = 100
    start_eval = 410

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HeCo_'+own_str+'.pkl'))
    model.eval()
    os.remove('HeCo_'+own_str+'.pkl')
    embeds = model.get_embeds(feats, mps)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
        # 读取分类结果文件最后一行（格式：macro \t micro \t auc）
        try:
            cls_file = f"result_{args.dataset}{args.ratio[i]}.txt"
            with open(cls_file, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            last = lines[-1]
            parts = last.split("\t")
            macro = float(parts[0]); micro = float(parts[1]); auc = float(parts[2])
            print(f"[EvalSummary] dataset={args.dataset} seed={args.seed} ratio={args.ratio[i]} macro_f1={macro:.4f} micro_f1={micro:.4f} auc={auc:.4f}")
        except Exception as e:
            print(f"[EvalSummary] dataset={args.dataset} seed={args.seed} ratio={args.ratio[i]} parse_error={e}")
    if (epoch + 1) % period == 0 and epoch > args.start_eval:
        print("---------------------------------------------------")
        run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset, epoch + 1)
    # 新增：训练结束后强制执行一次聚类评估，确保结果写出
    try:
        print("---------------------------------------------------")
        print("[Info] Running final clustering evaluation to ensure results are written...")
        final_epoch_tag = best_t + 1  # 用最佳epoch作为标记
        run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset, final_epoch_tag)
        print(f"[Info] Clustering results written to result_{args.dataset}_NMI&ARI.txt (Epoch: {final_epoch_tag})")
    except Exception as e:
        print(f"[Warning] Final clustering evaluation failed: {e}")
        # 读取聚类结果文件最后一行（格式：time \t Epoch: E \t NMI: xx \t ARI: yy）
        try:
            clu_file = f"result_{args.dataset}_NMI&ARI.txt"
            with open(clu_file, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            last = lines[-1]
            tokens = last.replace("\t", " ").split()
            # 例：['2025-10-13', '12:00', 'Epoch:', '500', 'NMI:', '66.12', 'ARI:', '58.03']
            def _get_after(tok):
                for j in range(len(tokens)-1):
                    if tokens[j] == tok:
                        return float(tokens[j+1])
                return None
            nmi = _get_after("NMI:")
            ari = _get_after("ARI:")
            print(f"[ClusterSummary] dataset={args.dataset} seed={args.seed} epoch={epoch+1} NMI={nmi:.2f} ARI={ari:.2f}")
        except Exception as e:
            print(f"[ClusterSummary] dataset={args.dataset} seed={args.seed} epoch={epoch+1} parse_error={e}")
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()
