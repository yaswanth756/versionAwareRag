"""
Category 3: Code-Example Questions (30 Questions)
Focus: How-to implement specific patterns, algorithms, concurrency paradigms, and standard library recipes.
"""

QUESTIONS = [
    {
        "query_id": "Q061",
        "question": "How can an iterable be chunked into fixed-length batches using itertools.batched() in Python 3.12?",
        "category": "code_example",
        "expected_answer": "In Python 3.12, itertools.batched(iterable, n) batches elements into tuples of length n. For example: list(batched('ABCDEFG', 3)) produces [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)].",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_itertools::api_itertools.batched::0",
                "relevance_score": 3,
                "evidence_text": "flattened_data"
            }
        ]
    },
    {
        "query_id": "Q062",
        "question": "How do you define a generator-based context manager using @contextlib.contextmanager?",
        "category": "code_example",
        "expected_answer": "Decorate a generator function with @contextlib.contextmanager, perform acquisition setup before yield, yield the resource, and perform teardown in a finally block.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_contextlib::api_contextlib.contextmanager::0",
                "relevance_score": 3,
                "evidence_text": "managed_resource"
            }
        ]
    },
    {
        "query_id": "Q063",
        "question": "How do you create an INET streaming socket in Python using the socket module?",
        "category": "code_example",
        "expected_answer": "Call s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) to create a TCP streaming socket with IPv4 addressing.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_sockets::sec_creating-a-socket::1",
                "relevance_score": 3,
                "evidence_text": "socket.socket"
            }
        ]
    },
    {
        "query_id": "Q064",
        "question": "How can you sort a list of tuples by a specific index using a key function in Python?",
        "category": "code_example",
        "expected_answer": "Pass a lambda or key function to sorted(data, key=lambda student: student[2]) or list.sort(key=...) to extract the sorting comparison key from each element.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_sorting::sec_key-functions::0",
                "relevance_score": 3,
                "evidence_text": "student_tuples"
            }
        ]
    },
    {
        "query_id": "Q065",
        "question": "How do you set up a basic command-line argument parser with argparse?",
        "category": "code_example",
        "expected_answer": "Import argparse, instantiate parser = argparse.ArgumentParser(), add arguments with parser.add_argument(), and invoke args = parser.parse_args().",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_argparse::sec_the-basics::0",
                "relevance_score": 3,
                "evidence_text": "parser = argparse.ArgumentParser()"
            }
        ]
    },
    {
        "query_id": "Q066",
        "question": "How do you configure logging using a dictionary schema with logging.config.dictConfig?",
        "category": "code_example",
        "expected_answer": "Define a dictionary containing version: 1, formatters, handlers, and loggers, then call logging.config.dictConfig(config_dict).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_logging-cookbook::sec_custom-handling-of-levels::2",
                "relevance_score": 3,
                "evidence_text": "logging.config.dictConfig"
            }
        ]
    },
    {
        "query_id": "Q067",
        "question": "How do you recursively walk a directory tree and inspect files using pathlib.Path.walk()?",
        "category": "code_example",
        "expected_answer": "Import Path from pathlib, create a root Path object, and iterate over for dirpath, dirnames, filenames in p.walk(): to traverse the directory hierarchy.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_pathlib::api_pathlib.Path.walk::1",
                "relevance_score": 3,
                "evidence_text": "from pathlib import Path"
            }
        ]
    },
    {
        "query_id": "Q068",
        "question": "How do you retain references to background asyncio tasks to avoid garbage collection?",
        "category": "code_example",
        "expected_answer": "Maintain a strong reference in a collection: background_tasks = set(); task = asyncio.create_task(coro()); background_tasks.add(task); task.add_done_callback(background_tasks.discard).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_asyncio-task::api_asyncio.create_task::0",
                "relevance_score": 3,
                "evidence_text": "background_tasks = set()"
            }
        ]
    },
    {
        "query_id": "Q069",
        "question": "How do you memoize recursive functions like factorial using functools.cache?",
        "category": "code_example",
        "expected_answer": "Apply @functools.cache directly above the recursive function definition (e.g., def factorial(n): ...), which caches previous return values indefinitely without maxsize limits.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functools::api_functools.cache::0",
                "relevance_score": 3,
                "evidence_text": "factorial"
            }
        ]
    },
    {
        "query_id": "Q070",
        "question": "How do you define a basic data class with type annotations using @dataclass?",
        "category": "code_example",
        "expected_answer": "Import dataclass from dataclasses, decorate the class with @dataclass, and define fields with type annotations (e.g. x: int, y: float = 0.0) to auto-generate __init__, __repr__, and __eq__.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_dataclasses::api_dataclasses.dataclass::3",
                "relevance_score": 3,
                "evidence_text": "class C:"
            }
        ]
    },
    {
        "query_id": "Q071",
        "question": "How do you implement a priority queue or heapsort using heapq.heappush and heapq.heappop?",
        "category": "code_example",
        "expected_answer": "Initialize an empty list h = [], insert elements using heapq.heappush(h, item), and extract the smallest elements in ascending order with heapq.heappop(h).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_heapq::sec_basic-examples::0",
                "relevance_score": 3,
                "evidence_text": "heappop(h)"
            }
        ]
    },
    {
        "query_id": "Q072",
        "question": "How do you write structured dictionaries into a CSV file with headers using csv.DictWriter?",
        "category": "code_example",
        "expected_answer": "Instantiate writer = csv.DictWriter(csvfile, fieldnames=['first_name', 'last_name']), call writer.writeheader(), and write rows using writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'}).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_csv::api_csv.DictWriter::0",
                "relevance_score": 3,
                "evidence_text": "csv.DictWriter"
            }
        ]
    },
    {
        "query_id": "Q073",
        "question": "How can you group elements into a list of items sharing the same key using collections.defaultdict?",
        "category": "code_example",
        "expected_answer": "Create d = defaultdict(list) and append items with d[key].append(value). When a key is first accessed, an empty list is automatically created.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_collections::sec_defaultdict-examples::0",
                "relevance_score": 3,
                "evidence_text": "defaultdict(list)"
            }
        ]
    },
    {
        "query_id": "Q074",
        "question": "How do you copy a directory tree while ignoring specific file extensions using shutil.copytree and shutil.ignore_patterns?",
        "category": "code_example",
        "expected_answer": "Call shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pyc', 'tmp*')) to recursively copy directories while skipping matching patterns.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_shutil::sec_copytree-example::0",
                "relevance_score": 3,
                "evidence_text": "copytree"
            }
        ]
    },
    {
        "query_id": "Q075",
        "question": "How can you remove a directory tree with error handling using shutil.rmtree?",
        "category": "code_example",
        "expected_answer": "Use shutil.rmtree(path, onexc=...) to delete an entire directory tree, providing an exception handler callback for permission or I/O failures.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_shutil::sec_rmtree-example::0",
                "relevance_score": 3,
                "evidence_text": "rmtree"
            }
        ]
    },
    {
        "query_id": "Q076",
        "question": "How can you generate random floating point numbers and sample items with random.random()?",
        "category": "code_example",
        "expected_answer": "Call random.random() to get a float uniformly distributed in the semi-open range [0.0, 1.0).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_random::sec_examples::0",
                "relevance_score": 3,
                "evidence_text": "random()"
            }
        ]
    },
    {
        "query_id": "Q077",
        "question": "How do you compute the sum of products of two numeric vectors using math.sumprod in Python 3.12?",
        "category": "code_example",
        "expected_answer": "In Python 3.12, call math.sumprod(p, q) with two iterables of equal length to calculate the dot product sum(p_i * q_i for p_i, q_i in zip(p, q)).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_math::api_math.sumprod::0",
                "relevance_score": 3,
                "evidence_text": "sumprod"
            }
        ]
    },
    {
        "query_id": "Q078",
        "question": "How do you implement an efficient FIFO queue using collections.deque instead of a list?",
        "category": "code_example",
        "expected_answer": "Use queue = deque(['Eric', 'John', 'Michael']), enqueue with queue.append('Terry'), and dequeue with queue.popleft() which runs in O(1) time.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_datastructures::sec_using-lists-as-queues::0",
                "relevance_score": 3,
                "evidence_text": "deque"
            }
        ]
    },
    {
        "query_id": "Q079",
        "question": "How do you attach a custom signal handler for SIGALRM or SIGINT using the signal module?",
        "category": "code_example",
        "expected_answer": "Define a handler function handler(signum, frame) and register it using signal.signal(signal.SIGALRM, handler).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_signal::sec_examples::0",
                "relevance_score": 3,
                "evidence_text": "signal.signal"
            }
        ]
    },
    {
        "query_id": "Q080",
        "question": "How do you extract multiple fields from dictionaries or tuples using operator.itemgetter?",
        "category": "code_example",
        "expected_answer": "Instantiate f = operator.itemgetter(1, 3) and call f(data), which fetches elements at indices 1 and 3 as a tuple.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_operator::api_operator.itemgetter::0",
                "relevance_score": 3,
                "evidence_text": "itemgetter"
            }
        ]
    },
    {
        "query_id": "Q081",
        "question": "How do you create and write files into a ZIP archive using zipfile.ZipFile?",
        "category": "code_example",
        "expected_answer": "Open an archive with with zipfile.ZipFile('archive.zip', 'w') as zf: and call zf.write('filename.txt') or zf.writestr('virtual.txt', 'data') to store files.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_zipfile::api_zipfile.ZipFile::2",
                "relevance_score": 3,
                "evidence_text": "ZipFile"
            }
        ]
    },
    {
        "query_id": "Q082",
        "question": "How do you execute SQL queries and fetch rows using sqlite3 cursor objects?",
        "category": "code_example",
        "expected_answer": "Call cur = con.cursor(), execute commands with cur.execute('SELECT * FROM stocks WHERE symbol=?', (symbol,)), and retrieve rows using cur.fetchall() or cur.fetchone().",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_sqlite3::api_sqlite3.Cursor::2",
                "relevance_score": 3,
                "evidence_text": "cur.execute"
            }
        ]
    },
    {
        "query_id": "Q083",
        "question": "How do you stream large JSON structures piece-by-piece using json.JSONEncoder.iterencode?",
        "category": "code_example",
        "expected_answer": "Iterate over for chunk in json.JSONEncoder().iterencode(bigobject): and write each string chunk to a socket or file buffer incrementally.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_json::api_json.JSONEncoder::1",
                "relevance_score": 3,
                "evidence_text": "for chunk in json.JSONEncoder().iterencode(bigobject):"
            }
        ]
    },
    {
        "query_id": "Q084",
        "question": "How do you parse a timestamp string into a datetime object using datetime.datetime.strptime?",
        "category": "code_example",
        "expected_answer": "Call datetime.strptime(date_string, format) passing standard directives like '%Y-%m-%d %H:%M:%S'.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_datetime::api_datetime.datetime.strptime::0",
                "relevance_score": 3,
                "evidence_text": "strptime"
            }
        ]
    },
    {
        "query_id": "Q085",
        "question": "How do you execute tasks concurrently across worker threads with concurrent.futures.ThreadPoolExecutor?",
        "category": "code_example",
        "expected_answer": "Use with ThreadPoolExecutor(max_workers=5) as executor: and call executor.map(func, iterable) or executor.submit(func, *args) to schedule concurrent jobs.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_concurrent.futures::api_concurrent.futures.Executor::1",
                "relevance_score": 3,
                "evidence_text": "ThreadPoolExecutor"
            }
        ]
    },
    {
        "query_id": "Q086",
        "question": "How do you create a named temporary file that auto-deletes on close using tempfile.NamedTemporaryFile?",
        "category": "code_example",
        "expected_answer": "Use with tempfile.NamedTemporaryFile() as fp: fp.write(b'Hello world!'); fp.seek(0); print(fp.read()). The file is deleted automatically upon exiting the context.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_tempfile::sec_examples::0",
                "relevance_score": 3,
                "evidence_text": "NamedTemporaryFile"
            }
        ]
    },
    {
        "query_id": "Q087",
        "question": "How do you mock a class or function during unit tests using @unittest.mock.patch?",
        "category": "code_example",
        "expected_answer": "Decorate the test function with @patch('module.TargetClass') and accept the mocked object as a parameter in the test method definition.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_unittest.mock::sec_patch::0",
                "relevance_score": 3,
                "evidence_text": "@patch('__main__.SomeClass')"
            }
        ]
    },
    {
        "query_id": "Q088",
        "question": "How do you take the first n elements of an iterable according to the itertools recipes?",
        "category": "code_example",
        "expected_answer": "Implement the take(n, iterable) recipe: def take(n, iterable): return list(islice(iterable, n)).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_itertools::sec_itertools-recipes::0",
                "relevance_score": 3,
                "evidence_text": "def take(n, iterable):"
            }
        ]
    },
    {
        "query_id": "Q089",
        "question": "How do you consume or advance an iterator n steps without storing the items in memory?",
        "category": "code_example",
        "expected_answer": "Implement the consume(iterator, n=None) recipe using collections.deque(islice(iterator, n), maxlen=0).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_itertools::sec_itertools-recipes::1",
                "relevance_score": 3,
                "evidence_text": "def consume(iterator, n=None):"
            }
        ]
    },
    {
        "query_id": "Q090",
        "question": "How do you define a function that prints a Fibonacci series up to n in Python?",
        "category": "code_example",
        "expected_answer": "Define def fib(n): a, b = 0, 1; while a < n: print(a, end=' '); a, b = b, a+b; print().",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_controlflow::sec_defining-functions::1",
                "relevance_score": 3,
                "evidence_text": "def fib(n):"
            }
        ]
    }
]

