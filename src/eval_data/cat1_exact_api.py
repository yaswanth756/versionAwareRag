"""
Category 1: Exact API Lookup (30 Questions)
Focus: Specific signatures, parameters, default values, return types, and documented behaviors.
"""

QUESTIONS = [
    {
        "query_id": "Q001",
        "question": "What parameters does itertools.batched() accept and how does it handle elements when the input length is not divisible by the batch size?",
        "category": "exact_api_lookup",
        "expected_answer": "itertools.batched(iterable, n) accepts an iterable and an integer batch size n. It batches data from the iterable into tuples of length n, and the last batch may be shorter than n if the total number of items is not evenly divisible by n.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_itertools::api_itertools.batched::0",
                "relevance_score": 3,
                "evidence_text": "Batch data from the iterable into tuples of length n. The last batch may be shorter than n."
            }
        ]
    },
    {
        "query_id": "Q002",
        "question": "What are the default parameter values and the return type for pathlib.Path.walk()?",
        "category": "exact_api_lookup",
        "expected_answer": "Path.walk(top_down=True, on_error=None, follow_symlinks=False) defaults to top_down=True, on_error=None, and follow_symlinks=False. It generates file and directory names by walking the directory tree, yielding a 3-tuple of (dirpath, dirnames, filenames).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_pathlib::api_pathlib.Path.walk::0",
                "relevance_score": 3,
                "evidence_text": "Generate the file names in a directory tree by walking the tree either top-down or bottom-up."
            }
        ]
    },
    {
        "query_id": "Q003",
        "question": "How should the typing.override decorator be used and what are type checkers expected to do when it is applied?",
        "category": "exact_api_lookup",
        "expected_answer": "typing.override is a decorator used to indicate that a method in a subclass is intended to override a method or attribute in a superclass. Type checkers should emit an error if the decorated method does not actually override anything in any base class.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_typing::api_typing.override::0",
                "relevance_score": 3,
                "evidence_text": "Decorator to indicate that a method in a subclass is intended to override a method or attribute in a superclass."
            }
        ]
    },
    {
        "query_id": "Q004",
        "question": "What is the default maxsize for functools.lru_cache and what happens if maxsize is set to None?",
        "category": "exact_api_lookup",
        "expected_answer": "functools.lru_cache has a default maxsize of 128. If maxsize is set to None, the LRU feature is disabled and the cache can grow without bound.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functools::api_functools.lru_cache::0",
                "relevance_score": 3,
                "evidence_text": "Decorator to wrap a function with a memoizing callable that saves up to the maxsize most recent calls."
            }
        ]
    },
    {
        "query_id": "Q005",
        "question": "What are the default tolerances in math.isclose(a, b)?",
        "category": "exact_api_lookup",
        "expected_answer": "In math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0), rel_tol defaults to 1e-09 (relative tolerance) and abs_tol defaults to 0.0 (minimum absolute tolerance).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_math::api_math.isclose::0",
                "relevance_score": 3,
                "evidence_text": "Return True if the values a and b are close to each other and False otherwise."
            }
        ]
    },
    {
        "query_id": "Q006",
        "question": "How does json.dumps treat dictionary keys that are not strings?",
        "category": "exact_api_lookup",
        "expected_answer": "Keys in key/value pairs of JSON are always of type str. When a dictionary is converted into JSON using json.dumps, all keys are coerced to strings, meaning loads(dumps(x)) != x if x has non-string keys.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_json::api_json.dumps::0",
                "relevance_score": 3,
                "evidence_text": "Serialize obj to a JSON formatted str using this conversion table."
            }
        ]
    },
    {
        "query_id": "Q007",
        "question": "Why is dataclasses.MISSING used as a sentinel in dataclasses.field() instead of None?",
        "category": "exact_api_lookup",
        "expected_answer": "dataclasses.MISSING is a sentinel object used because None is a valid default value for fields with a distinct meaning. Specifying both default and default_factory in field() is an error.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_dataclasses::api_dataclasses.field::0",
                "relevance_score": 3,
                "evidence_text": "To satisfy this need for additional information, you can replace the default field value with a call to the provided field() function."
            }
        ]
    },
    {
        "query_id": "Q008",
        "question": "What parameter in shutil.copytree controls whether an existing destination directory raises an error or is merged into?",
        "category": "exact_api_lookup",
        "expected_answer": "The dirs_exist_ok parameter (defaulting to False). If dirs_exist_ok is True, copytree will not raise a FileExistsError if dst already exists, and will copy files into the existing tree.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_shutil::api_shutil.copytree::0",
                "relevance_score": 3,
                "evidence_text": "Recursively copy an entire directory tree rooted at src to a directory named dst and return the destination directory."
            }
        ]
    },
    {
        "query_id": "Q009",
        "question": "What new parameter was added to sqlite3.connect() in Python 3.12 to manage transaction control according to PEP 670?",
        "category": "exact_api_lookup",
        "expected_answer": "Python 3.12 added the autocommit parameter to sqlite3.connect(), controlling transaction management and conforming to PEP 670.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_sqlite3::api_sqlite3.connect::0",
                "relevance_score": 3,
                "evidence_text": "Open a connection to an SQLite database."
            }
        ]
    },
    {
        "query_id": "Q010",
        "question": "What does contextlib.nullcontext return from its __enter__ method when initialized with enter_result?",
        "category": "exact_api_lookup",
        "expected_answer": "contextlib.nullcontext(enter_result=None) returns enter_result from its __enter__() method, while otherwise performing no operation.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_contextlib::api_contextlib.nullcontext::0",
                "relevance_score": 3,
                "evidence_text": "Return a context manager that returns enter_result from __enter__, but otherwise does nothing."
            }
        ]
    },
    {
        "query_id": "Q011",
        "question": "What is collections.Counter and what method returns the n most frequent elements?",
        "category": "exact_api_lookup",
        "expected_answer": "collections.Counter is a dict subclass for counting hashable objects. The most_common([n]) method returns a list of the n most common elements and their counts from the most common to the least.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_collections::api_collections.Counter::0",
                "relevance_score": 3,
                "evidence_text": "Counter is a dict subclass for counting hashable objects."
            }
        ]
    },
    {
        "query_id": "Q012",
        "question": "How are tasks spawned within an asyncio.TaskGroup?",
        "category": "exact_api_lookup",
        "expected_answer": "Tasks are added to an asyncio.TaskGroup using the tg.create_task() method within an async with TaskGroup() context.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_asyncio-task::api_asyncio.TaskGroup::0",
                "relevance_score": 3,
                "evidence_text": "An asynchronous context manager holding a group of tasks. Tasks can be added to the group using create_task()."
            }
        ]
    },
    {
        "query_id": "Q013",
        "question": "What does secrets.token_hex(nbytes=None) return and how many hex characters are in each byte?",
        "category": "exact_api_lookup",
        "expected_answer": "secrets.token_hex(nbytes=None) returns a random text string in hexadecimal. Each byte of randomness is converted to two hexadecimal digits.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_secrets::api_secrets.token_hex::0",
                "relevance_score": 3,
                "evidence_text": "Return a random text string, in hexadecimal."
            }
        ]
    },
    {
        "query_id": "Q014",
        "question": "What arguments does hashlib.file_digest() take and what mode must the file object be opened in?",
        "category": "exact_api_lookup",
        "expected_answer": "hashlib.file_digest(fileobj, digest, /) accepts a file-like object and a digest algorithm. The fileobj must be opened for reading in binary mode.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_hashlib::api_hashlib.file_digest::0",
                "relevance_score": 3,
                "evidence_text": "Return a digest object that has been updated with contents of file object."
            }
        ]
    },
    {
        "query_id": "Q015",
        "question": "What 5 components are returned by urllib.parse.urlsplit()?",
        "category": "exact_api_lookup",
        "expected_answer": "urllib.parse.urlsplit() returns a 5-item named tuple (SplitResult) with attributes: scheme, netloc, path, query, and fragment.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_urllib.parse::api_urllib.parse.urlsplit::0",
                "relevance_score": 3,
                "evidence_text": "This function returns a 5-item named tuple:"
            }
        ]
    },
    {
        "query_id": "Q016",
        "question": "How does inspect.iscoroutinefunction() determine if an object is a coroutine function?",
        "category": "exact_api_lookup",
        "expected_answer": "inspect.iscoroutinefunction(object) returns True if the object is a coroutine function created with an async def syntax.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_inspect::api_inspect.iscoroutinefunction::0",
                "relevance_score": 3,
                "evidence_text": "Return True if the object is a coroutine function"
            }
        ]
    },
    {
        "query_id": "Q017",
        "question": "How does bisect.bisect_right differ from bisect.bisect_left when encountering existing equal entries?",
        "category": "exact_api_lookup",
        "expected_answer": "bisect.bisect_right returns an insertion point that comes after (to the right of) any existing matching entries in a sorted sequence, whereas bisect_left returns an insertion point before (to the left of) existing entries.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_bisect::api_bisect.bisect_right::0",
                "relevance_score": 3,
                "evidence_text": "Similar to bisect_left(), but returns an insertion point which comes after"
            }
        ]
    },
    {
        "query_id": "Q018",
        "question": "What is the default mode and compression method for zipfile.ZipFile?",
        "category": "exact_api_lookup",
        "expected_answer": "zipfile.ZipFile(file, mode='r', compression=ZIP_STORED, ...) defaults to mode='r' (read-only) and compression=ZIP_STORED (uncompressed).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_zipfile::api_zipfile.ZipFile::0",
                "relevance_score": 3,
                "evidence_text": "Open a ZIP file, where file can be a path to a file"
            }
        ]
    },
    {
        "query_id": "Q019",
        "question": "What is csv.DictReader and what happens if a row in the CSV has more fields than fieldnames?",
        "category": "exact_api_lookup",
        "expected_answer": "csv.DictReader creates an object that maps row data to a dictionary where field keys come from the first row or the fieldnames parameter. If a row has more fields than fieldnames, the remaining data is stored in a list keyed by the restkey parameter (defaulting to None).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_csv::api_csv.DictReader::0",
                "relevance_score": 3,
                "evidence_text": "Create an object that operates like a regular reader but maps the information in each row to a dict"
            }
        ]
    },
    {
        "query_id": "Q020",
        "question": "How does random.choices differ from random.sample in terms of replacement?",
        "category": "exact_api_lookup",
        "expected_answer": "random.choices chooses elements from the population with replacement (elements can be chosen multiple times), whereas random.sample samples without replacement.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_random::api_random.choices::0",
                "relevance_score": 3,
                "evidence_text": "Return a k sized list of elements chosen from the population with replacement."
            }
        ]
    },
    {
        "query_id": "Q021",
        "question": "What is string.Template.substitute() and what happens when placeholders are missing?",
        "category": "exact_api_lookup",
        "expected_answer": "string.Template.substitute(mapping={}, /, **kwds) performs template substitution and raises a KeyError if a placeholder is missing from mapping and kwds. In contrast, safe_substitute() leaves missing placeholders intact.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_string::api_string.Template::0",
                "relevance_score": 3,
                "evidence_text": "Performs the template substitution, returning a new string."
            }
        ]
    },
    {
        "query_id": "Q022",
        "question": "What does os.scandir() yield and why is it preferred over os.listdir() for file attribute inspection?",
        "category": "exact_api_lookup",
        "expected_answer": "os.scandir(path='.') yields an iterator of os.DirEntry objects. It is faster than os.listdir() because DirEntry objects cache file attributes without requiring additional os.stat() system calls on supported OSes.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_os::api_os.scandir::0",
                "relevance_score": 3,
                "evidence_text": "Return an iterator of os.DirEntry objects corresponding to the entries in the directory given by path."
            }
        ]
    },
    {
        "query_id": "Q023",
        "question": "What does the ignore_cleanup_errors parameter do in tempfile.TemporaryDirectory?",
        "category": "exact_api_lookup",
        "expected_answer": "When ignore_cleanup_errors=True, any exceptions encountered while cleaning up and deleting the directory tree upon exit are ignored rather than raising an error.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_tempfile::api_tempfile.TemporaryDirectory::0",
                "relevance_score": 3,
                "evidence_text": "This class securely creates a temporary directory using the same rules as mkdtemp()."
            }
        ]
    },
    {
        "query_id": "Q024",
        "question": "What does statistics.fmean() calculate and how does its speed compare to statistics.mean()?",
        "category": "exact_api_lookup",
        "expected_answer": "statistics.fmean(data, weights=None) converts data to floats and computes the arithmetic mean. It runs faster than the mean() function and always returns a float.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_statistics::api_statistics.fmean::0",
                "relevance_score": 3,
                "evidence_text": "Convert data to floats and compute the arithmetic mean."
            }
        ]
    },
    {
        "query_id": "Q025",
        "question": "What does heapq.merge() do and what must be true about its inputs?",
        "category": "exact_api_lookup",
        "expected_answer": "heapq.merge(*iterables, key=None, reverse=False) merges multiple sorted inputs into a single sorted iterator. The input iterables must already be sorted according to key and reverse.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_heapq::api_heapq.merge::0",
                "relevance_score": 3,
                "evidence_text": "Merge multiple sorted inputs into a single sorted output"
            }
        ]
    },
    {
        "query_id": "Q026",
        "question": "What does logging.basicConfig() do when handlers are already attached to the root logger?",
        "category": "exact_api_lookup",
        "expected_answer": "By default, logging.basicConfig() does nothing if the root logger already has handlers configured, unless the force=True keyword argument is provided (added in Python 3.8).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_logging::api_logging.basicConfig::0",
                "relevance_score": 3,
                "evidence_text": "Does basic configuration for the logging system by creating a StreamHandler with a default Formatter"
            }
        ]
    },
    {
        "query_id": "Q027",
        "question": "What formats does datetime.datetime.fromisoformat() parse in Python 3.11 and 3.12?",
        "category": "exact_api_lookup",
        "expected_answer": "In Python 3.11+, datetime.fromisoformat(date_string) parses any valid ISO 8601 format (including time zone offsets with fractional seconds and 'T' replaced by any single separator), unlike earlier versions which only parsed formats emitted by isoformat().",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_datetime::api_datetime.datetime.fromisoformat::0",
                "relevance_score": 3,
                "evidence_text": "Return a datetime corresponding to a date_string in any valid ISO 8601 format"
            }
        ]
    },
    {
        "query_id": "Q028",
        "question": "What is enum.StrEnum and how does it compare to regular Enum?",
        "category": "exact_api_lookup",
        "expected_answer": "enum.StrEnum is a subclass of Enum whose members are also strings (subclasses of str). StrEnum members compare directly with other strings using standard equality.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_enum::api_enum.StrEnum::0",
                "relevance_score": 3,
                "evidence_text": "StrEnum is the same as Enum, but its members are also strings"
            }
        ]
    },
    {
        "query_id": "Q029",
        "question": "What does uuid.uuid4() generate and what class is returned?",
        "category": "exact_api_lookup",
        "expected_answer": "uuid.uuid4() generates a random UUID (version 4) using os.urandom(), returning a uuid.UUID object instance.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_uuid::api_uuid.uuid4::0",
                "relevance_score": 3,
                "evidence_text": "Generate a random UUID."
            }
        ]
    },
    {
        "query_id": "Q030",
        "question": "What clock does time.perf_counter() use and what is its primary use case?",
        "category": "exact_api_lookup",
        "expected_answer": "time.perf_counter() returns the value of a monotonic performance counter with the highest available resolution to measure short durations (benchmarking). It is unaffected by system clock updates.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_time::api_time.perf_counter::0",
                "relevance_score": 3,
                "evidence_text": "Return the value (in fractional seconds) of a performance counter"
            }
        ]
    }
]

