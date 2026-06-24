"""Re-shard parquet: file gốc 3.6GB/1 row-group -> reader nạp NGUYÊN row-group = OOM.
Ghi lại với row_group nhỏ -> HF streaming nạp từng nhúm dòng, RAM phẳng.
Chạy 1 lần: python reshard.py  (đọc Python/, ghi Python_rg/). Xong đổi SHARDS sang Python_rg/.
"""
import glob, os
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/workspace/data/math-dataset/Python/*.parquet"
DST = "/workspace/data/math-dataset/Python_rg"
RG = 256  # dòng/row-group: nhỏ đủ để 1 nhúm ảnh vừa RAM, đủ to để IO không phí

os.makedirs(DST, exist_ok=True)
for src in sorted(glob.glob(SRC)):
    out = os.path.join(DST, os.path.basename(src))
    if os.path.exists(out):
        print("skip", out); continue
    pf = pq.ParquetFile(src)
    w = None
    # iter_batches đọc từng RG dòng trong row-group khổng lồ mà KO nạp cả file
    for batch in pf.iter_batches(batch_size=RG):
        t = pa.Table.from_batches([batch])
        if w is None:
            w = pq.ParquetWriter(out, t.schema)
        w.write_table(t, row_group_size=RG)
    w.close()
    print("done", out)
print("OK -> đổi SHARDS sang", DST, "trong train_stage*.py")
