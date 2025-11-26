# ======================================================
# WPP HYBRID ENGINE v1
# PART 1 — CONFIG, LOADING, BASIC PREPROCESSING
# ======================================================

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

import warnings
warnings.filterwarnings("ignore")


# ======================================================
# CONFIG (ALL CONFIGURABLE VIA FLAGS)
# ======================================================

DATA_PATH = "wpp_2020_2024_final.parquet"
CATEGORY_MAP_PATH = "catmap.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===== Model Parameters =====
MF_FACTORS = 64
MF_REG = 0.1
MF_ITERS = 20

TOP_K_RECS = 10
TEMPORAL_SPLIT_RATIO = 0.75

# ===== Hybrid Weights =====
WEIGHT_ALS = 0.50       # MF latent factors
WEIGHT_IBCF = 0.30      # item similarity
WEIGHT_CAT = 0.10       # category embeddings
WEIGHT_RECENCY = 0.05   # recent merchants
WEIGHT_POP = 0.05       # popularity boost

# ===== Filtering =====
MIN_INTERACTIONS = 8
MAX_INTERACTIONS = None    # None = disabled

# ===== Diversification =====
ENABLE_DIVERSITY = True
DIVERSITY_RULE = "CAP_PER_CATEGORY"   # CAP_PER_CATEGORY or RATIO_PERSONAL_EXPLORE

CATEGORY_CAP = 2                      # max items per category
EXPLORE_RATIO = 0.3                   # 30 percent exploration

# ===== Clustering =====
ENABLE_CLUSTERING = True
AUTO_K_RANGE = (2, 10)        # min=2, max=10 clusters
AUTO_K_SAMPLE_SIZE = 5000

# ===== Evaluation =====
EVAL_USER_SAMPLE_SIZE = 8000


# ======================================================
# LOAD DATA
# ======================================================

print("\n📥 Loading parquet file...")
usecols = [
    "member_id_str",
    "is_anonymous",
    "merchant",
    "createdAt",
    "categories",
    "name",
    "year",
]

df = pd.read_parquet(DATA_PATH, columns=usecols)
print(f"Loaded {len(df):,} rows")

# basic cleanup
df = df.dropna(subset=["merchant", "createdAt", "member_id_str"])
df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
df["merchant"] = df["merchant"].astype(int)

# ensure categories is list of ints
df["categories"] = df["categories"].apply(
    lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
)
df["categories"] = df["categories"].apply(lambda x: [int(c) for c in x])

# member only for ALS
is_member = ~df["is_anonymous"].astype(bool)
df_members = df[is_member].copy()

# filter by minimum interactions
member_counts = df_members["member_id_str"].value_counts()
active_members = member_counts[member_counts >= MIN_INTERACTIONS].index
df_members = df_members[df_members["member_id_str"].isin(active_members)]

if MAX_INTERACTIONS is not None:
    mc = df_members["member_id_str"].value_counts()
    good_members = mc[mc <= MAX_INTERACTIONS].index
    df_members = df_members[df_members["member_id_str"].isin(good_members)]

print(f"Active members: {df_members['member_id_str'].nunique():,}")
print(f"Remaining member interactions: {len(df_members):,}")

# merchant → name mapping
merchant_name_map = (
    df[["merchant", "name"]]
    .dropna()
    .drop_duplicates(subset=["merchant"])
    .set_index("merchant")["name"]
    .to_dict()
)

# load category id → name map
catmap = pd.read_csv(CATEGORY_MAP_PATH)
category_id_to_name = dict(zip(catmap["id"], catmap["name"]))

print(f"Loaded {len(category_id_to_name)} categories from catmap.csv")


# ======================================================
# PART 2 — ALS + IBCF + CATEGORY EMBEDDING + RECENCY + POP + SESSION BOOSTS
# ======================================================

# ------------------------------------------------------
# 1. TEMPORAL SPLIT
# ------------------------------------------------------
print("\n🕒 Performing temporal split...")

df_members = df_members.sort_values("createdAt")

cutoff_idx = int(len(df_members) * TEMPORAL_SPLIT_RATIO)
df_train = df_members.iloc[:cutoff_idx].copy()
df_test  = df_members.iloc[cutoff_idx:].copy()

print(f"Train interactions: {len(df_train):,}")
print(f"Test interactions:  {len(df_test):,}")

# ------------------------------------------------------
# 2. ENCODE USERS AND ITEMS
# ------------------------------------------------------
print("\n🔢 Encoding user and merchant ids...")

user_ids = df_members["member_id_str"].unique()
item_ids = df_members["merchant"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
idx_to_user = {i: u for u, i in user_to_idx.items()}

item_to_idx = {m: i for i, m in enumerate(item_ids)}
idx_to_item = {i: m for m, i in item_to_idx.items()}

df_train["u_idx"] = df_train["member_id_str"].map(user_to_idx)
df_train["i_idx"] = df_train["merchant"].map(item_to_idx)

df_test["u_idx"] = df_test["member_id_str"].map(user_to_idx)
df_test["i_idx"] = df_test["merchant"].map(item_to_idx)

num_users = len(user_ids)
num_items = len(item_ids)

print(f"Users: {num_users:,}, Merchants: {num_items:,}")

# ------------------------------------------------------
# 3. BUILD USER–ITEM MATRICES
# ------------------------------------------------------
print("\n📐 Building CSR matrices...")

train_ui = csr_matrix(
    (np.ones(len(df_train), dtype=np.float32),
     (df_train["u_idx"], df_train["i_idx"])),
    shape=(num_users, num_items)
)

test_ui = csr_matrix(
    (np.ones(len(df_test), dtype=np.float32),
     (df_test["u_idx"], df_test["i_idx"])),
    shape=(num_users, num_items)
)

print(f"Train UI shape: {train_ui.shape}, Test UI shape: {test_ui.shape}")

# ======================================================
#                     ALS (BM25)
# ======================================================
print("\n⚙️ ALS training (with BM25 weighting)...")

item_user = train_ui.T.tocsr()        # shape: (num_items, num_users)
item_user_bm25 = bm25_weight(item_user, K1=100, B=0.8)

als = AlternatingLeastSquares(
    factors=MF_FACTORS,
    regularization=MF_REG,
    iterations=MF_ITERS,
    random_state=RANDOM_STATE
)

als.fit(item_user_bm25)

MF_ITEMS = als.user_factors      # (num_items, factors)
MF_USERS = als.item_factors      # (num_users, factors)

print("ALS completed.")
print("MF_USERS shape:", MF_USERS.shape)
print("MF_ITEMS shape:", MF_ITEMS.shape)

# ======================================================
#             IBCF SIMILARITY (BM25 WEIGHTED)
# ======================================================
print("\n🧩 Computing BM25-weighted item–item similarity for IBCF...")

item_user_bm25_dense = item_user_bm25.tocsr()

similarity_matrix = cosine_similarity(
    item_user_bm25_dense, item_user_bm25_dense
).astype(np.float32)

np.fill_diagonal(similarity_matrix, 0.0)
item_sim_sparse = csr_matrix(similarity_matrix)

print("IBCF similarity built.")

# ======================================================
#         SESSION-AWARE CO-VISIT BOOST 
# ======================================================
print("\n🔗 Building session-aware co-visit matrix...")

# We assume df_train already has session_id column prepared in preprocessing
if "session_id" in df_members.columns:
    df_sess = df_train[["session_id", "merchant"]].copy()
    co_counts = np.zeros((num_items, num_items), dtype=np.float32)

    for s, grp in df_sess.groupby("session_id"):
        items = grp["merchant"].unique()
        idxs = [item_to_idx[m] for m in items if m in item_to_idx]
        for i in idxs:
            for j in idxs:
                if i != j:
                    co_counts[i, j] += 1

    # normalize
    if co_counts.max() > 0:
        co_counts = co_counts / co_counts.max()
else:
    co_counts = np.zeros((num_items, num_items), dtype=np.float32)

session_sim = csr_matrix(co_counts)
print("Co-visit session similarity ready.")

# ======================================================
#              CATEGORY EMBEDDINGS (MULTI-HOT)
# ======================================================
print("\n🏷 Building category embeddings...")

merchant_categories = defaultdict(list)

for r in df[["merchant", "categories"]].drop_duplicates("merchant").itertuples():
    merchant = r.merchant
    cats = r.categories if isinstance(r.categories, list) else []
    merchant_categories[merchant] = cats

all_cats = set()
for cat_list in merchant_categories.values():
    all_cats.update(cat_list)
num_categories = max(all_cats) + 1 if len(all_cats) > 0 else 1

print(f"Detected {num_categories} total category IDs")

cat_matrix = np.zeros((num_items, num_categories), dtype=np.float32)

for iid in range(num_items):
    mid = idx_to_item[iid]
    for c in merchant_categories.get(mid, []):
        if 0 <= c < num_categories:
            cat_matrix[iid, c] = 1.0

row_sums = cat_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cat_matrix = cat_matrix / row_sums

print("Category multi-hot embedding matrix ready:", cat_matrix.shape)

# ======================================================
#              RECENCY SCORES
# ======================================================
print("\n⏳ Computing recency scores with short-term boost...")

last_ts = df_train.groupby("merchant")["createdAt"].max()
global_max = last_ts.max()
age_days = (global_max - last_ts).dt.total_seconds() / (3600 * 24)

# long-term decay
recency_raw = np.exp(-0.05 * age_days)

# ---- NEW: short-term 48-hour boost ----
boost_cut = global_max - pd.Timedelta(hours=48)
recent_mask = last_ts >= boost_cut
recency_raw[recent_mask] *= 1.3   # configurable boost factor

recency_scores = recency_raw.reindex(item_ids).fillna(0).values.astype(np.float32)
print("Example recency:", recency_scores[:10])

# ======================================================
#              POPULARITY 
# ======================================================
print("\n📊 Computing improved popularity scores...")

item_pop = df_train["merchant"].value_counts().reindex(item_ids).fillna(0).values

# log-pop to reduce dominance
item_pop_log = np.log1p(item_pop)

# normalize
item_pop_norm = item_pop_log / (item_pop_log.max() + 1e-9)
item_pop_norm = item_pop_norm.astype(np.float32)

print("Example popularity:", item_pop_norm[:10])


# ======================================================
# PART 3 — FINAL HYBRID RECOMMENDER + DIVERSIFICATION
# ======================================================

# ------------------------------------------------------
# 1. Hybrid Scoring Configuration (default ON)
# ------------------------------------------------------
ENABLE_ALS = True
ENABLE_IBCF = True
ENABLE_SESSIONS = True
ENABLE_CATEGORY = True
ENABLE_POP = True
ENABLE_RECENCY = True
ENABLE_DIVERSITY = True
ENABLE_EXPOSURE_PENALTY = False     # optional, default OFF

# weights (all configurable)
W_ALS = 0.50
W_IBCF = 0.20
W_SESS = 0.10         
W_CAT = 0.10
W_POP = 0.15
W_REC = 0.05

# diversity
MAX_SAME_MERCHANT = 1
MAX_PER_CATEGORY = 3
EXPLORATION_RATIO = 0.20


# ------------------------------------------------------
# 2. Utility — Scoring Functions
# ------------------------------------------------------

def safe_norm(x):
    """Normalize array to 0-1 safely."""
    x = x.astype(np.float32)
    mx = x.max()
    if mx > 0:
        return x / mx
    return x


def get_ibcf_scores(u_idx):
    """IBCF: BM25 item-item similarity."""
    u_vec = train_ui[u_idx]
    if u_vec.nnz == 0:
        return np.zeros(num_items, dtype=np.float32)

    scores = u_vec.dot(item_sim_sparse).toarray().ravel()
    return safe_norm(scores)


def get_session_scores(u_idx):
    """Session-based co-visit similarity."""
    if not ENABLE_SESSIONS:
        return np.zeros(num_items, dtype=np.float32)

    u_vec = train_ui[u_idx]
    if u_vec.nnz == 0:
        return np.zeros(num_items, dtype=np.float32)

    scores = u_vec.dot(session_sim).toarray().ravel()
    return safe_norm(scores)


def get_category_scores():
    """Item–item similarity in multi-hot category space."""
    cat_norm = (cat_matrix @ cat_matrix.T)
    cat_norm = np.clip(cat_norm, 0, 1)
    return cat_norm


# precompute category similarity
if ENABLE_CATEGORY:
    print("\n📦 Precomputing category similarity matrix...")
    CAT_SIM = get_category_scores()
else:
    CAT_SIM = None


def get_category_signal(u_idx):
    """User preference for categories."""
    if CAT_SIM is None:
        return np.zeros(num_items, dtype=np.float32)

    u_vec = train_ui[u_idx].toarray().ravel()
    if u_vec.sum() == 0:
        return np.zeros(num_items, dtype=np.float32)

    scores = u_vec @ CAT_SIM
    return safe_norm(scores)


# ------------------------------------------------------
# 3. FINAL HYBRID SCORE
# ------------------------------------------------------
def hybrid_score(u_idx):
    """Compute final hybrid score for user u_idx."""

    final = np.zeros(num_items, dtype=np.float32)

    # ALS latent factors
    if ENABLE_ALS:
        als_s = MF_USERS[u_idx] @ MF_ITEMS.T
        final += W_ALS * safe_norm(als_s)

    # IBCF similarity
    if ENABLE_IBCF:
        final += W_IBCF * get_ibcf_scores(u_idx)

    # Session co-visit
    if ENABLE_SESSIONS:
        final += W_SESS * get_session_scores(u_idx)

    # Category embedding signal
    if ENABLE_CATEGORY:
        final += W_CAT * get_category_signal(u_idx)

    # Popularity
    if ENABLE_POP:
        final += W_POP * item_pop_norm

    # Recency
    if ENABLE_RECENCY:
        final += W_REC * recency_scores

    # Optional: exposure penalty (items user saw too often)
    if ENABLE_EXPOSURE_PENALTY:
        penalty = train_ui[u_idx].toarray().ravel()
        if penalty.sum() > 0:
            penalty = penalty / penalty.max()
            final -= 0.05 * penalty    # configurable
            final = np.clip(final, 0, None)

    return final


# ------------------------------------------------------
# 4. DIVERSITY ENFORCER 
# ------------------------------------------------------
def apply_diversity(ranked_items):
    """Apply merchant and category caps."""

    if not ENABLE_DIVERSITY:
        return list(ranked_items[:TOP_K_RECS])

    final = []
    merchant_count = Counter()
    category_count = Counter()

    def get_item_cats(iid):
        mid = idx_to_item[iid]
        return merchant_categories.get(mid, [])

    for iid in ranked_items:
        mid = idx_to_item[iid]
        cats = get_item_cats(iid)

        # rule 1: merchant cap
        if merchant_count[mid] >= MAX_SAME_MERCHANT:
            continue

        # rule 2: category cap
        if any(category_count[c] >= MAX_PER_CATEGORY for c in cats):
            continue

        # accept item
        final.append(iid)
        merchant_count[mid] += 1
        for c in cats:
            category_count[c] += 1

        if len(final) >= TOP_K_RECS:
            break

    # fill missing with remaining
    if len(final) < TOP_K_RECS:
        need = TOP_K_RECS - len(final)
        filler = [x for x in ranked_items if x not in final][:need]
        final.extend(filler)

    return final


# ------------------------------------------------------
# 5. FINAL RECOMMENDER FUNCTION
# ------------------------------------------------------
def recommend_items_for_user(u_idx, top_n=TOP_K_RECS):
    """Main hybrid recommender."""

    scores = hybrid_score(u_idx)
    ranked = np.argsort(scores)[::-1]
    diversified = apply_diversity(ranked)

    return diversified[:top_n]


# ======================================================
# PART 4 — EVALUATION + AUTO-K CLUSTERING + DEMO 
# ======================================================

import pandas as pd
from tqdm import tqdm
from collections import Counter

print("\n=======================================================")
print("📊 PART 4 — Evaluation + Auto-K Clustering + Demo")
print("=======================================================\n")


# ------------------------------------------------------
# 1. METRICS
# ------------------------------------------------------
def precision_at_k(pred, truth, k):
    if len(truth) == 0:
        return 0.0
    return len(set(pred[:k]) & set(truth)) / k


def map_at_k(pred, truth, k):
    truth = set(truth)
    if len(truth) == 0:
        return 0.0

    hits = 0
    s = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in truth:
            hits += 1
            s += hits / i
    return s / min(len(truth), k)


def ndcg_at_k(pred, truth, k):
    truth = set(truth)
    if len(truth) == 0:
        return 0.0

    dcg = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in truth:
            dcg += 1.0 / np.log2(i + 1)

    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ------------------------------------------------------
# 2. GLOBAL EVALUATION
# ------------------------------------------------------
def evaluate_model(user_indices=None, verbose=True):
    """Evaluate the hybrid recommender."""
    if user_indices is None:
        user_indices = np.arange(num_users)

    rng = np.random.default_rng(42)
    # sample users if too many
    if len(user_indices) > EVAL_USER_SAMPLE_SIZE:
        user_indices = rng.choice(
            user_indices, size=EVAL_USER_SAMPLE_SIZE, replace=False
        )

    p_list, m_list, n_list = [], [], []

    for u in tqdm(user_indices, disable=not verbose):
        truth = test_ui[u].indices
        if len(truth) == 0:
            continue

        pred = recommend_items_for_user(u, top_n=TOP_K_RECS)

        p_list.append(precision_at_k(pred, truth, TOP_K_RECS))
        m_list.append(map_at_k(pred, truth, TOP_K_RECS))
        n_list.append(ndcg_at_k(pred, truth, TOP_K_RECS))

    return {
        "Precision@K": float(np.mean(p_list)),
        "MAP@K": float(np.mean(m_list)),
        "NDCG@K": float(np.mean(n_list)),
    }


print("\n🔍 Evaluating Hybrid Model...")
global_metrics = evaluate_model()

print("\n📊 GLOBAL EVALUATION RESULTS:")
for k, v in global_metrics.items():
    print(f"{k}: {v:.4f}")


# ------------------------------------------------------
# 3. AUTO-K CLUSTERING
# ------------------------------------------------------
if ENABLE_CLUSTERING:

    print("\n🤖 Running Auto-K clustering...")

    X = MF_USERS  # ALS user embeddings

    # sample for silhouette search
    rng = np.random.default_rng(42)
    if len(X) > AUTO_K_SAMPLE_SIZE:
        idx_sample = rng.choice(len(X), size=AUTO_K_SAMPLE_SIZE, replace=False)
        X_sample = X[idx_sample]
    else:
        X_sample = X

    best_k = None
    best_score = -1
    Ks = list(range(AUTO_K_RANGE[0], AUTO_K_RANGE[1] + 1))

    for k in Ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)

        if score > best_score:
            best_k = k
            best_score = score

    print(f"📌 Best K automatically selected: {best_k}  (silhouette={best_score:.4f})")

    # final clustering
    kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(MF_USERS)

    # evaluate per cluster
    cluster_results = []
    for cid in range(best_k):
        users = np.where(cluster_labels == cid)[0]
        eval_users = [u for u in users if test_ui[u].nnz > 0]

        if len(eval_users) == 0:
            continue

        metrics = evaluate_model(
            user_indices=np.array(eval_users), verbose=False
        )

        cluster_results.append({
            "cluster": cid,
            "n_eval_users": len(eval_users),
            "Precision@K": metrics["Precision@K"],
            "MAP@K": metrics["MAP@K"],
            "NDCG@K": metrics["NDCG@K"],
        })

    print("\n📊 PER-CLUSTER RESULTS:")
    print(pd.DataFrame(cluster_results).round(4))

# ======================================================
# PART 5 — ADVANCED DEMO
# ======================================================

print("\n=======================================================")
print("🎛️ PART 5 — Demo Tools")
print("=======================================================\n")

# ------------------------------------------------------
# 1. Helper — merchant name lookup
# ------------------------------------------------------
def mid_to_name(mid):
    """Return readable merchant name."""
    return merchant_name_map.get(mid, f"Merchant_{mid}")


# ------------------------------------------------------
# 2. Helper — retrieve user cluster ID
# ------------------------------------------------------
def get_user_cluster(u_idx):
    if "cluster_labels" not in globals():
        return None
    try:
        return int(cluster_labels[u_idx])
    except Exception:
        return None


# ------------------------------------------------------
# 3. Helper — compute category distribution
# ------------------------------------------------------
def compute_category_profile(item_indices):
    """Return category proportion dictionary for a list of item indices."""
    profile = Counter()
    total = 0

    for iid in item_indices:
        mid = idx_to_item[iid]
        cats = merchant_categories.get(mid, [])
        for c in cats:
            profile[c] += 1
            total += 1

    if total == 0:
        return {}

    return {c: round(v / total, 3) for c, v in profile.items()}


# ------------------------------------------------------
# 4. Pretty Demo
# ------------------------------------------------------
def pretty_demo(member_id_str, top_n=TOP_K_RECS):

    print("\n" + "="*90)
    print(f"👤 USER DEMO — {member_id_str}")
    print("="*90)

    # Lookup
    if member_id_str not in user_to_idx:
        print(f"❌ Member {member_id_str} not found.")
        return

    u = user_to_idx[member_id_str]

    # Train & Test behaviour
    train_items = train_ui[u].indices
    test_items  = test_ui[u].indices

    train_names = [mid_to_name(idx_to_item[i]) for i in train_items]
    test_names  = [mid_to_name(idx_to_item[i]) for i in test_items]

    # cluster assignment
    user_cluster = get_user_cluster(u)

    # Recommendations
    rec_items = recommend_items_for_user(u, top_n=top_n)
    rec_names = [mid_to_name(idx_to_item[i]) for i in rec_items]

    # metrics
    prec = precision_at_k(rec_items, test_items, top_n)
    mapk = map_at_k(rec_items, test_items, top_n)
    ndcg = ndcg_at_k(rec_items, test_items, top_n)

    # category profiles
    train_cat_prof = compute_category_profile(train_items)
    rec_cat_prof   = compute_category_profile(rec_items)

    # -------------------------
    # PRINTING
    # -------------------------

    # cluster
    if user_cluster is not None:
        print(f"\n📌 Cluster: {user_cluster}")
    else:
        print("\n📌 Cluster: (clustering disabled)")

    # train
    print("\n📜 USER PAST BEHAVIOUR (train):")
    print(" ", ", ".join(train_names[:25]) or "(none)")

    # test
    print("\n📅 RECENT ACTIVITY (test):")
    print(" ", ", ".join(test_names) or "(none)")

    # recommendations
    print("\n🤖 MODEL RECOMMENDS:")
    for i, nm in enumerate(rec_names, 1):
        print(f"  {i}. {nm}")

    # eval
    print("\n📊 EVALUATION:")
    print(f"  Precision@{top_n}: {prec:.3f}")
    print(f"  MAP@{top_n}:       {mapk:.3f}")
    print(f"  NDCG@{top_n}:      {ndcg:.3f}")

    # category distribution
    print("\n🏷 CATEGORY PROFILE — TRAIN:")
    print(train_cat_prof or "(none)")

    print("\n🏷 CATEGORY PROFILE — RECOMMENDATIONS:")
    print(rec_cat_prof or "(none)")

    print("="*90 + "\n")


# ------------------------------------------------------
# 5. Demo user selection utility
# ------------------------------------------------------
def pick_demo_users(min_train=5, max_train=20, min_test=3):
    """
    Automatically selects meaningful demo users with:
      - enough TRAIN items to show preferences
      - enough TEST items so metrics are not zero
    """
    good = []
    for u in range(num_users):
        t = len(train_ui[u].indices)
        r = len(test_ui[u].indices)

        if min_train <= t <= max_train and r >= min_test:
            good.append(idx_to_user[u])

    return good[:10]


print("🎯 Demo tools loaded. Use:")
print("    pretty_demo('member_id_str')")
print("Or auto-select demo-friendly users with:")
print("    pick_demo_users()")

# ======================================================
# EXPORT GLOBALS FOR STREAMLIT DASHBOARD
# ======================================================

# basic global variables
num_users = num_users
num_items = num_items
df_train = df_train
df_test = df_test

# popularity and recency
recency_scores = recency_scores
item_pop_norm = item_pop_norm

# category mappings
merchant_categories = merchant_categories
category_id_to_name = category_id_to_name
cat_matrix = cat_matrix

# ALS embeddings
MF_USERS = MF_USERS
MF_ITEMS = MF_ITEMS

# similarity matrices
item_sim_sparse = item_sim_sparse
CAT_SIM = CAT_SIM

# user and item maps
user_to_idx = user_to_idx
idx_to_user = idx_to_user
item_to_idx = item_to_idx
idx_to_item = idx_to_item

# training and test CSR matrices
train_ui = train_ui
test_ui = test_ui

# cluster labels (optional)
if 'cluster_labels' in globals():
    cluster_labels = cluster_labels

# global evaluation (optional)
if 'global_metrics' in globals():
    global_metrics = global_metrics
