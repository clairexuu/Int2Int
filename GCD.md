# 1-Layer GCD Training Command

# TODO
take the embedding and do
- fourier transform
- svd
- pca

## Command

@dumped/gcd_1layer/1

```bash
python train.py --dump_path ./dumped --exp_name gcd_1layer --exp_id 1 \
    --operation gcd \
    --n_enc_layers 1 --n_dec_layers 1 \
    --enc_emb_dim 128 --dec_emb_dim 128 \
    --n_enc_heads 4 --n_dec_heads 4 \
    --gelu_activation false \
    --max_epoch 200 \
    --epoch_size 50000 \
    --batch_size 64 \
    --optimizer "adam,lr=0.0001" \
    --maxint 100000
```

@dumped/gcd_1layer/2 

```bash
python train.py --dump_path ./dumped --exp_name gcd_1layer --exp_id 2 \
    --operation gcd \
    --n_enc_layers 0 --n_dec_layers 1 \
    --enc_emb_dim 128 --dec_emb_dim 128 \
    --n_enc_heads 4 --n_dec_heads 4 \
    --gelu_activation false \
    --max_epoch 200 \
    --epoch_size 50000 \
    --batch_size 128 \
    --optimizer "adamw,lr=0.0001,beta1=0.9,beta2=0.98,weight_decay=1" \
    --maxint 113 \
    --cpu true
```

@dumped/gcd_grok/1

Step 1: Generate fixed 30%/30% train/eval split (mirrors on-the-fly encoding from train.py)

```bash
python generate_gcd_data.py
```

Step 2: Train with fixed data for grokking (25,000 data passes, eval every 100 passes = 250 evals)

```bash
python train.py --dump_path ./dumped --exp_name gcd_grok --exp_id 1 \
    --operation gcd \
    --n_enc_layers 0 --n_dec_layers 1 \
    --enc_emb_dim 128 --dec_emb_dim 128 \
    --n_enc_heads 4 --n_dec_heads 4 \
    --gelu_activation false \
    --max_epoch 250 \
    --epoch_size 383000 \
    --batch_size 128 \
    --optimizer "adamw,lr=0.0001,beta1=0.9,beta2=0.98,weight_decay=1" \
    --maxint 113 \
    --train_data data/gcd_train.txt \
    --eval_data "data/gcd_eval.txt,data/gcd_eval.txt" \
    --eval_size 3830 \
    --num_workers 0 \
    --batch_size_eval 128
```

@dumped/gcd_grok/2

Step 1: Generate fixed 30%/30% train/eval split (mirrors on-the-fly encoding from train.py)

```bash
python generate_gcd_data.py
```

Step 2: Train with fixed data for grokking (25,000 data passes, eval every 100 passes = 250 evals)

```bash
python train.py --dump_path ./dumped --exp_name gcd_grok --exp_id 2 \
    --operation gcd \
    --n_enc_layers 0 --n_dec_layers 1 \
    --enc_emb_dim 128 --dec_emb_dim 128 \
    --n_enc_heads 4 --n_dec_heads 4 \
    --gelu_activation false \
    --max_epoch 250 \
    --epoch_size 383000 \
    --batch_size 128 \
    --optimizer "adamw,lr=0.001,beta1=0.9,beta2=0.98,weight_decay=1" \
    --maxint 113 \
    --train_data data/gcd_train.txt \
    --eval_data "data/gcd_eval.txt,data/gcd_eval.txt" \
    --eval_size 3830 \
    --num_workers 0 \
    --batch_size_eval 128
```


## Parameters

### Task
| Parameter | Value | Description |
|---|---|---|
| `--operation gcd` | `gcd` | Train the model to compute GCD of two integers |
| `--maxint` | `100000` | Input integers sampled uniformly from [1, 100000] |

### Architecture
| Parameter | Value | Description |
|---|---|---|
| `--n_enc_layers` | `1` | 1 Transformer layer in the encoder |
| `--n_dec_layers` | `1` | 1 Transformer layer in the decoder |
| `--enc_emb_dim` | `128` | Token embedding dimension d = 128 (encoder) |
| `--dec_emb_dim` | `128` | Token embedding dimension d = 128 (decoder) |
| `--n_enc_heads` | `4` | 4 attention heads in encoder, each of dimension d/4 = 32 |
| `--n_dec_heads` | `4` | 4 attention heads in decoder, each of dimension d/4 = 32 |
| `--gelu_activation` | `false` | Use ReLU activation in the FFN (not GELU) |
| MLP hidden dim | `512` | Automatically computed as `dim * 4 = 128 * 4 = 512` (hardcoded in `src/model/transformer.py:404`) |

### Defaults (not in command, but active)
| Parameter | Default | Description |
|---|---|---|
| `--sinusoidal_embeddings` | `false` | Uses learned positional embeddings (not sinusoidal) |
| `--enc_has_pos_emb` | `true` | Encoder has positional embeddings |
| `--dec_has_pos_emb` | `true` | Decoder has positional embeddings |
| `--share_inout_emb` | `true` | Input and output embeddings are shared |
| `--dropout` | `0` | No dropout (fine since data is generated on-the-fly) |
| `--base` | `1000` | Integers encoded in base 1000 |
| `--clip_grad_norm` | `5` | Gradient clipping norm |

### Training
| Parameter | Value | Description |
|---|---|---|
| `--max_epoch` | `200` | Train for up to 200 epochs (sufficient for a 1-layer model to converge or plateau) |
| `--epoch_size` | `50000` | Generate 50k training examples per epoch before evaluating (more frequent eval than the 300k default) |
| `--batch_size` | `64` | 64 examples per batch (larger than default 32 since the model is small) |
| `--optimizer` | `adam,lr=0.0001` | Adam optimizer with learning rate 1e-4 |

### Output
| Parameter | Value | Description |
|---|---|---|
| `--dump_path` | `./dumped` | Save experiment logs and checkpoints here |
| `--exp_name` | `gcd_1layer` | Experiment name for organization |
| `--exp_id` | `1` | Experiment run ID |
