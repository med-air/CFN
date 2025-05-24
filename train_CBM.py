import numpy as np
import random
import argparse
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle
from sklearn.linear_model import OrthogonalMatchingPursuit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="ham")
    parser.add_argument('--trial', default="")
    parser.add_argument('--backbone', default="densenet")
    parser.add_argument('--num_concept', type=int, default=800)
    parser.add_argument('--result_dir', default="")
    args = parser.parse_args()

    img_feat = np.load(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/img_emb_train.npy").astype(np.double)
    concept_feat = np.load(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/concept_cav.npy").astype(np.double)
    img_feat_val = np.load(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/img_emb_val.npy")
    img_feat_test = np.load(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/img_emb_test.npy")
    with open(f"{args.result_dir}/{args.data}/target.pkl", "rb") as f:
        train_label, val_label, test_label = pickle.load(f)

    out = np.load(f"{args.result_dir}/{args.data}/{args.backbone}/{args.trial}/select_idx_{args.num_concept}.npy")
    X = concept_feat[out].T
    

    weights = []
    img_emb_train_fit = []
    for i in tqdm(range(len(img_feat))):
        y = img_feat[i]
        reg = OrthogonalMatchingPursuit(n_nonzero_coefs=160).fit(X, y)
        img_emb_train_fit.append(reg.predict(X).reshape(1, -1))
    img_emb_train_fit = np.concatenate(img_emb_train_fit, axis=0)

    weights = []
    img_emb_val_fit = []
    for i in tqdm(range(len(img_feat_val))):
        y = img_feat_val[i]
        reg = OrthogonalMatchingPursuit(n_nonzero_coefs=160).fit(X, y)
        img_emb_val_fit.append(reg.predict(X).reshape(1, -1))
    img_emb_val_fit = np.concatenate(img_emb_val_fit, axis=0)

    weights = []
    img_emb_test_fit = []
    for i in tqdm(range(len(img_feat_test))):
        y = img_feat_test[i]
        reg = OrthogonalMatchingPursuit(n_nonzero_coefs=160).fit(X, y)
        img_emb_test_fit.append(reg.predict(X).reshape(1, -1))
    img_emb_test_fit = np.concatenate(img_emb_test_fit, axis=0)

    train_features, val_features, test_features = img_emb_train_fit, img_emb_val_fit, img_emb_test_fit
    train_labels, val_labels, test_labels = train_label, val_label, test_label


    val_acc_step_list = np.zeros([3, 8])
    best_c_weights_list = []
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        random.seed(seed)
        search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
        acc_list = []
        for c_weight in search_list:
            clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight).fit(train_features,
                                                                                                  train_labels)
            pred = clf.predict(val_features)
            acc_val = np.mean([int(t == p) for t, p in zip(val_labels, pred)]).astype(np.float32) * 100.
            acc_list.append(acc_val)

        print(acc_list, flush=True)

        # binary search
        peak_idx = np.argmax(acc_list)
        c_peak = search_list[peak_idx]
        c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak


        def binary_search(c_left, c_right, seed, step, val_acc_step_list):
            clf_left = LogisticRegression(  # random_state=0,
                C=c_left,
                max_iter=1000,
                verbose=0,
                n_jobs=4)
            clf_left.fit(train_features, train_labels)
            pred_left = clf_left.predict(val_features)
            accuracy = np.mean((val_labels == pred_left).astype(np.float32)) * 100.
            acc_left = np.mean([int(t == p) for t, p in zip(val_labels, pred_left)]).astype(np.float32) * 100
            print("Val accuracy (Left): {:.2f}".format(acc_left), flush=True)

            clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right).fit(train_features,
                                                                                                       train_labels)
            pred_right = clf_right.predict(val_features)
            acc_right = np.mean([int(t == p) for t, p in zip(val_labels, pred_right)]).astype(np.float32) * 100
            print("Val accuracy (Right): {:.2f}".format(acc_right), flush=True)

            # find maximum and update ranges
            if acc_left < acc_right:
                c_final = c_right
                clf_final = clf_right
                # range for the next step
                c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_right = np.log10(c_right)
            else:
                c_final = c_left
                clf_final = clf_left
                # range for the next step
                c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
                c_left = np.log10(c_left)

            pred = clf_final.predict(val_features)
            val_acc = np.mean([int(t == p) for t, p in zip(val_labels, pred)]).astype(np.float32) * 100
            print("Val Accuracy: {:.2f}".format(val_acc), flush=True)
            val_acc_step_list[seed - 1, step] = val_acc

            # saveline = "{}, seed {}, {} shot, weight {}, val_acc {:.2f}\n".format(cfg.dataset, seed, cfg.n_shots, c_final, val_acc)
            return (
                np.power(10, c_left),
                np.power(10, c_right),
                seed,
                step,
                val_acc_step_list,
            )


        for step in range(8):
            c_left, c_right, seed, step, val_acc_step_list = binary_search(c_left, c_right, seed, step, val_acc_step_list)

        # save c_left as the optimal weight for each run
        best_c_weights_list.append(c_left)


    best_c = np.mean(best_c_weights_list)
    classifier = LogisticRegression(random_state=0,
                                    C=best_c,
                                    max_iter=1000,
                                    verbose=0)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    print(np.mean(predictions==test_labels))