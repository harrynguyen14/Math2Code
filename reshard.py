"""Re-shard: gốc 20 file x 205k dòng/3.6GB (1 row-group -> OOM khi nạp).
Chia lại thành shard ~50k dòng -> mỗi shard ~0.9GB, load THẲNG vào RAM (non-streaming)
nên train random-access được -> bật group_by_length (gom mẫu cùng độ dài, VRAM phẳng).
Data đã trộn ngẫu nhiên nên mỗi shard nhỏ vẫn lẫn đủ loại.

Chạy 1 lần: python reshard.py  (Python/ -> Python_rg/). Xong đổi SHARDS sang Python_rg/.
"""
import glob, os
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/workspace/data/math-dataset/Python/*.parquet"
DST = "/workspace/data/math-dataset/Python_rg"
PER_SHARD = 50_000  # dòng/shard out: ~0.9GB ảnh, vừa RAM 64GB khi load 1 shard

os.makedirs(DST, exist_ok=True)
n_rows, shard_i, w = 0, 0, None

def flush_writer():  # đóng file shard hiện tại
    global w
    if w is not None:
        w.close(); w = None

for src in sorted(glob.glob(SRC)):
    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(batch_size=2048):  # đọc nhúm nhỏ, ko nạp cả file 3.6GB
        t = pa.Table.from_batches([batch])
        if w is None:
            out = os.path.join(DST, f"shard-{shard_i:05d}.parquet")
            w = pq.ParquetWriter(out, t.schema)
        w.write_table(t)
        n_rows += t.num_rows
        if n_rows >= PER_SHARD:        # đủ 1 shard -> sang file mới
            flush_writer()
            print(f"done shard-{shard_i:05d} ({n_rows} dòng)")
            shard_i += 1; n_rows = 0
flush_writer()
if n_rows:
    print(f"done shard-{shard_i:05d} ({n_rows} dòng, shard cuối)")
print(f"OK -> {shard_i + (1 if n_rows else 0)} shard ở {DST}. Đổi SHARDS sang {DST}/*.parquet")
