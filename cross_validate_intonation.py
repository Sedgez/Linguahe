import os, gc, random, re
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sentence_transformers import SentenceTransformer

DATASET_PATH = 'intonation_dataset.csv'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
OUTPUT_DIR = os.path.join('intonation_ai', 'models', 'cross_validation')
N_SPLITS = 5
SEED = 42
ALPHA = 1.5
ACOUSTIC_COLUMNS = [
    'normalized_final_pitch_change',
    'normalized_final_pitch_slope',
    'normalized_final_pitch_range',
]
SENTENCE_PARTICLES = [
    'ba','ha','nga','naman','pala','na','pa','lang','lamang','daw','raw',
    'din','rin','eh','diba','yata','talaga','ano'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED); torch.manual_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f'{DATASET_PATH} not found. Put the dataset beside this script.')

df = pd.read_csv(DATASET_PATH).dropna(subset=['transcript','intonation_label',*ACOUSTIC_COLUMNS]).reset_index(drop=True)
texts = df['transcript'].astype(str).tolist()
acoustic = df[ACOUSTIC_COLUMNS].astype(np.float32).values
encoder = LabelEncoder(); y = encoder.fit_transform(df['intonation_label'].astype(str))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('='*60); print('LINGUAHE INTONTATION 5-FOLD CROSS-VALIDATION'); print('='*60)
print(f'Device: {DEVICE}'); print(f'Rows: {len(df)}'); print(f'Classes: {dict(zip(encoder.classes_, np.bincount(y)))}')

print('\nLoading SBERT...')
sbert = SentenceTransformer(MODEL_NAME, device=DEVICE)
transformer = sbert[0].auto_model; tokenizer = sbert.tokenizer; transformer.eval()


def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())[:30]


def rule_matches(text):
    matches = set()
    for sentence in re.split(r'[.!?]+', text.lower()):
        words = re.findall(r"\b[\w']+\b", sentence)
        if words and words[-1] in SENTENCE_PARTICLES:
            matches.add(words[-1])
    return matches


def ra_sbert(text):
    tokens = tokenize(text)
    if not tokens: return np.zeros(384, dtype=np.float32)
    matched = rule_matches(text)
    encoded = tokenizer(' '.join(tokens), truncation=True, max_length=128, return_tensors='pt', padding=False)
    encoded = {k:v.to(DEVICE) for k,v in encoded.items()}
    with torch.no_grad(): hidden = transformer(**encoded).last_hidden_state[0]
    tok = tokenizer(' '.join(tokens), truncation=True, max_length=128, return_tensors=None)
    word_ids = tok.word_ids()
    vectors, weights = [], []
    current = None; current_vecs = []; current_weight = 1.0
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != current:
            if current_vecs:
                vectors.append(torch.stack(current_vecs).mean(0)); weights.append(current_weight)
            current = wid; current_vecs = []
            word = tokens[wid] if wid < len(tokens) else ''
            current_weight = ALPHA if word in matched else 1.0
        current_vecs.append(hidden[i])
    if current_vecs:
        vectors.append(torch.stack(current_vecs).mean(0)); weights.append(current_weight)
    if not vectors: return np.zeros(384, dtype=np.float32)
    v = torch.stack(vectors); w = torch.tensor(weights, dtype=v.dtype, device=v.device).unsqueeze(1)
    sent = (v*w).sum(0) / w.sum(); norm = torch.norm(sent)
    if norm > 0: sent = sent / norm
    return sent.cpu().numpy().astype(np.float32)

print('\nCreating 384-D Intonation RA-SBERT vectors...')
ra_vectors = np.asarray([ra_sbert(t) for t in texts], dtype=np.float32)
print(f'RA-SBERT matrix: {ra_vectors.shape}')


def compile_model(inputs, output):
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def acoustic_model():
    i = tf.keras.Input((3,), name='acoustic_input')
    x = tf.keras.layers.Dense(16, activation='relu')(i)
    x = tf.keras.layers.Dropout(.15)(x)
    return compile_model(i, tf.keras.layers.Dense(2, activation='softmax')(x))


def ra_model():
    i = tf.keras.Input((384,), name='ra_sbert_input')
    x = tf.keras.layers.Dense(32, activation='relu')(i)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(.20)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x); x = tf.keras.layers.Dropout(.15)(x)
    return compile_model(i, tf.keras.layers.Dense(2, activation='softmax')(x))


def fusion_model():
    si = tf.keras.Input((384,), name='ra_sbert_input'); ai = tf.keras.Input((3,), name='acoustic_input')
    x = tf.keras.layers.Dense(32, activation='relu', name='sbert_projection')(si)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(.20)(x)
    x = tf.keras.layers.Concatenate(name='fusion_35d')([x, ai])
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Dropout(.20)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x); x = tf.keras.layers.Dropout(.15)(x)
    return compile_model([si, ai], tf.keras.layers.Dense(2, activation='softmax')(x))


def fit(model, xtr, ytr, xv, yv):
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=4, min_lr=1e-6),
    ]
    model.fit(xtr, ytr, validation_data=(xv, yv), epochs=80, batch_size=16, callbacks=cb, verbose=0)
    return model

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rows = []

for fold, (tr, va) in enumerate(skf.split(ra_vectors, y), 1):
    print('\n'+'='*60); print(f'FOLD {fold}/{N_SPLITS}'); print('='*60)
    a_scaler = StandardScaler().fit(acoustic[tr])
    atr = a_scaler.transform(acoustic[tr]).astype(np.float32); ava = a_scaler.transform(acoustic[va]).astype(np.float32)
    str_, sva = ra_vectors[tr], ra_vectors[va]; ytr, yva = y[tr], y[va]

    am = fit(acoustic_model(), atr, ytr, ava, yva)
    ap = np.argmax(am.predict(ava, verbose=0), axis=1); aa = accuracy_score(yva, ap)
    print(f'Acoustic-only: {aa*100:.2f}%'); print(confusion_matrix(yva, ap)); rows.append([fold,'Acoustic-only',aa])

    rm = fit(ra_model(), str_, ytr, sva, yva)
    rp = np.argmax(rm.predict(sva, verbose=0), axis=1); ra = accuracy_score(yva, rp)
    print(f'RA-SBERT-only: {ra*100:.2f}%'); print(confusion_matrix(yva, rp)); rows.append([fold,'RA-SBERT-only',ra])

    fm = fit(fusion_model(), [str_,atr], ytr, [sva,ava], yva)
    fp = np.argmax(fm.predict([sva,ava], verbose=0), axis=1); fa = accuracy_score(yva, fp)
    print(f'Small Fusion: {fa*100:.2f}%'); print(confusion_matrix(yva, fp)); rows.append([fold,'Small Fusion',fa])

    del am, rm, fm; tf.keras.backend.clear_session(); gc.collect()

results = pd.DataFrame(rows, columns=['fold','model','accuracy'])
summary = results.groupby('model')['accuracy'].agg(['mean','std','min','max']).reset_index()
results_path = os.path.join(OUTPUT_DIR, 'intonation_5fold_results.csv')
summary_path = os.path.join(OUTPUT_DIR, 'intonation_5fold_summary.csv')
results.to_csv(results_path, index=False); summary.to_csv(summary_path, index=False)

print('\n'+'='*60); print('FINAL 5-FOLD CROSS-VALIDATION RESULTS'); print('='*60)
for _, r in summary.iterrows():
    print(f"{r['model']:<22} Mean: {r['mean']*100:6.2f}%  Std: {r['std']*100:6.2f}%  Range: {r['min']*100:.2f}% - {r['max']*100:.2f}%")
print('='*60)
print(f'Detailed results: {results_path}')
print(f'Summary:          {summary_path}')
print('Existing trained model files were not overwritten.')
