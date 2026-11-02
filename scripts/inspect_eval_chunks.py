import json

with open("data/processed/python-3.12/chunks.jsonl") as f:
    chunks = [json.loads(line) for line in f]
chunk_map = {c["chunk_id"]: c for c in chunks}

cat1_targets = [
    ("itertools.batched", "library/itertools.html", "tuples of length n"),
    ("pathlib.Path.walk", "library/pathlib.html", "top_down = True"),
    ("typing.override", "library/typing.html", "override a method in a superclass"),
    ("functools.lru_cache", "library/functools.html", "maxsize = 128"),
    ("math.isclose", "library/math.html", "rel_tol = 1e-09"),
    ("json.dumps", "library/json.html", "indent"),
    ("dataclasses.field", "library/dataclasses.html", "default_factory"),
    ("shutil.copytree", "library/shutil.html", "dirs_exist_ok = False"),
    ("sqlite3.connect", "library/sqlite3.html", "autocommit"),
    ("contextlib.nullcontext", "library/contextlib.html", "enter_result = None"),
    ("collections.Counter", "library/collections.html", "most_common"),
    ("asyncio.TaskGroup", "library/asyncio-task.html", "create_task"),
    ("secrets.token_hex", "library/secrets.html", "nbytes = None"),
    ("hashlib.file_digest", "library/hashlib.html", "fileobj , digest"),
    ("urllib.parse.urlsplit", "library/urllib.parse.html", "SplitResult"),
    ("inspect.iscoroutinefunction", "library/inspect.html", "Return True if the object is a coroutine function"),
    ("bisect.bisect_right", "library/bisect.html", "lo = 0 , hi = None"),
    ("zipfile.ZipFile", "library/zipfile.html", "compression = 0"),
    ("csv.DictReader", "library/csv.html", "fieldnames = None"),
    ("random.choices", "library/random.html", "weights = None"),
    ("string.Template", "library/string.html", "safe_substitute"),
    ("os.scandir", "library/os.html", "DirEntry"),
    ("tempfile.TemporaryDirectory", "library/tempfile.html", "ignore_cleanup_errors = False"),
    ("statistics.fmean", "library/statistics.html", "weights = None"),
    ("heapq.merge", "library/heapq.html", "reverse = False"),
    ("logging.basicConfig", "library/logging.html", "encoding = None"),
    ("datetime.datetime.fromisoformat", "library/datetime.html", "date_string"),
    ("enum.StrEnum", "library/enum.html", "auto"),
    ("uuid.uuid4", "library/uuid.html", "Generate a random UUID"),
    ("time.perf_counter", "library/time.html", "highest available resolution"),
]

print("Validating Category 1 candidates:")
for sym, doc, sub in cat1_targets:
    found = False
    for cid, c in chunk_map.items():
        if c.get("relative_path") == doc and (c.get("symbol_name") == sym or (sym in (c.get("symbol_name") or ""))):
            if sub in c["searchable_text"]:
                print(f"  MATCH: {sym} -> {cid}")
                found = True
                break
    if not found:
        # search all chunks in doc
        for cid, c in chunk_map.items():
            if c.get("relative_path") == doc and sub in c["searchable_text"]:
                s = c.get("symbol_name")
                print(f"  FALLBACK DOC MATCH: {sym} -> {cid} (symbol={s})")
                found = True
                break
    if not found:
        print(f"  MISSING: {sym} ({sub})")

