"""
Category 6: Multi-Document / Comparison Questions (28 Questions)
Focus: Cross-document architectural comparisons between alternative modules, data structures, and concurrency models.
"""

QUESTIONS = [
    {
        "query_id": "Q147",
        "question": "How does pathlib.Path compare to os.path for filesystem path manipulations?",
        "category": "multi_document",
        "expected_answer": "pathlib.Path provides an object-oriented API representing filesystem paths as class instances with rich methods (/ operator, .stat(), .read_text()), whereas os.path provides low-level procedural functions operating directly on raw string or byte representations.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_pathlib::api_pathlib.Path::0",
                "relevance_score": 3,
                "evidence_text": "Path"
            },
            {
                "chunk_id": "library_os.path::api_os.path.join::0",
                "relevance_score": 2,
                "evidence_text": "Join one or more path segments"
            }
        ]
    },
    {
        "query_id": "Q148",
        "question": "What is the primary difference in execution model and memory space between threading.Thread and multiprocessing.Process?",
        "category": "multi_document",
        "expected_answer": "threading.Thread runs activities within the same OS process sharing memory space (constrained by the Global Interpreter Lock for bytecode execution), while multiprocessing.Process spawns independent OS processes with separate memory spaces and individual GILs.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_threading::api_threading.Thread::0",
                "relevance_score": 3,
                "evidence_text": "Thread"
            },
            {
                "chunk_id": "library_multiprocessing::api_multiprocessing.Process::0",
                "relevance_score": 3,
                "evidence_text": "Process"
            }
        ]
    },
    {
        "query_id": "Q149",
        "question": "When should asyncio tasks be used instead of OS threading for concurrent operations?",
        "category": "multi_document",
        "expected_answer": "asyncio tasks use cooperative multitasking on a single thread event loop, optimal for high-concurrency I/O-bound networking without OS thread context switching overhead, whereas threading is suited for blocking I/O calls without async library equivalents.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_asyncio-task::api_asyncio.create_task::0",
                "relevance_score": 3,
                "evidence_text": "Task"
            },
            {
                "chunk_id": "library_threading::api_threading.Thread::0",
                "relevance_score": 2,
                "evidence_text": "Thread"
            }
        ]
    },
    {
        "query_id": "Q150",
        "question": "What are the key trade-offs between json and pickle for object serialization?",
        "category": "multi_document",
        "expected_answer": "json serializes standard data structures into human-readable, cross-language text format with strict security guarantees. pickle serializes almost arbitrary Python objects into a binary Python-specific format, but is insecure against arbitrary code execution when unpickling untrusted data.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_json::api_json.dumps::0",
                "relevance_score": 3,
                "evidence_text": "JSON"
            },
            {
                "chunk_id": "library_pickle::api_pickle.dumps::0",
                "relevance_score": 3,
                "evidence_text": "pickle"
            }
        ]
    },
    {
        "query_id": "Q151",
        "question": "How do dataclasses compare with collections.namedtuple for record storage?",
        "category": "multi_document",
        "expected_answer": "dataclasses generate customizable classes that are mutable by default (unless frozen=True) and integrate inheritance and default_factory. namedtuple creates lightweight immutable tuple subclasses accessible by index or name with smaller memory overhead.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_dataclasses::api_dataclasses.dataclass::3",
                "relevance_score": 3,
                "evidence_text": "dataclass"
            },
            {
                "chunk_id": "library_collections::api_collections.namedtuple::0",
                "relevance_score": 3,
                "evidence_text": "namedtuple"
            }
        ]
    },
    {
        "query_id": "Q152",
        "question": "How does queue.Queue differ from multiprocessing.Queue in inter-thread vs inter-process communication?",
        "category": "multi_document",
        "expected_answer": "queue.Queue is an in-memory thread-safe FIFO queue synchronized via threading primitives for threads in the same process. multiprocessing.Queue uses OS pipes, buffers, and background feeder threads to serialize (pickle) and transfer objects across separate process boundaries.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_queue::api_queue.Queue::0",
                "relevance_score": 3,
                "evidence_text": "FIFO"
            },
            {
                "chunk_id": "library_multiprocessing::api_multiprocessing.Queue::0",
                "relevance_score": 3,
                "evidence_text": "queue"
            }
        ]
    },
    {
        "query_id": "Q153",
        "question": "What is the behavioral difference between list.sort() and the built-in sorted() function?",
        "category": "multi_document",
        "expected_answer": "list.sort() sorts a list in-place and returns None, modifying the original list. The built-in sorted() function accepts any iterable and returns a new sorted list leaving the source iterable unchanged.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_datastructures::sec_more-on-lists::0",
                "relevance_score": 3,
                "evidence_text": "sort"
            },
            {
                "chunk_id": "library_functions::api_sorted::0",
                "relevance_score": 3,
                "evidence_text": "sorted"
            }
        ]
    },
    {
        "query_id": "Q154",
        "question": "How does the 'is' identity comparison differ from the '==' equality operator in Python?",
        "category": "multi_document",
        "expected_answer": "The 'is' operator compares object identity (checking if two references point to the exact same memory address as verified by id()), whereas '==' evaluates value equality by invoking the __eq__() rich comparison method.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_expressions::sec_comparisons::0",
                "relevance_score": 3,
                "evidence_text": "Comparisons"
            },
            {
                "chunk_id": "library_functions::api_id::0",
                "relevance_score": 2,
                "evidence_text": "identity"
            }
        ]
    },
    {
        "query_id": "Q155",
        "question": "When should you prefer functools.partial over a lambda expression for function argument pre-binding?",
        "category": "multi_document",
        "expected_answer": "functools.partial creates a callable object that cleanly introspects arguments (.func, .args, .keywords), serializes via pickle, and avoids late-binding variable capture bugs common in lambdas inside loops.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functools::api_functools.partial::0",
                "relevance_score": 3,
                "evidence_text": "partial"
            },
            {
                "chunk_id": "tutorial_controlflow::sec_lambda-expressions::0",
                "relevance_score": 2,
                "evidence_text": "lambda"
            }
        ]
    },
    {
        "query_id": "Q156",
        "question": "How does parsing command-line parameters with argparse compare to manually reading sys.argv?",
        "category": "multi_document",
        "expected_answer": "sys.argv is simply the raw list of string arguments passed to the script. argparse provides structured flag parsing, automatic --help generation, type coercion, subcommands, and default values.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_sys::api_sys.argv::0",
                "relevance_score": 3,
                "evidence_text": "command line arguments"
            },
            {
                "chunk_id": "howto_argparse::sec_the-basics::0",
                "relevance_score": 3,
                "evidence_text": "ArgumentParser"
            }
        ]
    },
    {
        "query_id": "Q157",
        "question": "What is the difference in abstraction level between urllib.request and http.client?",
        "category": "multi_document",
        "expected_answer": "urllib.request is a high-level URL fetching library supporting protocols, redirects, proxies, and authentication. http.client is a low-level client implementing HTTP and HTTPS protocols directly.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_urllib.request::api_urllib.request.urlopen::0",
                "relevance_score": 3,
                "evidence_text": "urlopen"
            },
            {
                "chunk_id": "library_http.client::api_http.client.HTTPConnection::0",
                "relevance_score": 3,
                "evidence_text": "HTTPConnection"
            }
        ]
    },
    {
        "query_id": "Q158",
        "question": "How does itertools.zip_longest handle iterables of unequal length compared to built-in zip()?",
        "category": "multi_document",
        "expected_answer": "Built-in zip() stops yielding pairs when the shortest iterable is exhausted (or raises ValueError if strict=True). itertools.zip_longest continues until the longest iterable is exhausted, substituting a fillvalue (default None) for missing values.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_itertools::api_itertools.zip_longest::0",
                "relevance_score": 3,
                "evidence_text": "zip_longest"
            },
            {
                "chunk_id": "library_functions::api_zip::0",
                "relevance_score": 3,
                "evidence_text": "zip"
            }
        ]
    },
    {
        "query_id": "Q159",
        "question": "What is the functional difference between cryptographic hashing with hashlib and message authentication with hmac?",
        "category": "multi_document",
        "expected_answer": "hashlib provides cryptographic digests (like SHA-256) verifying data integrity without a secret key. hmac combines a cryptographic hash with a shared secret key to verify both data integrity and sender authenticity.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_hashlib::sec_hash-algorithms::0",
                "relevance_score": 3,
                "evidence_text": "sha256"
            },
            {
                "chunk_id": "library_hmac::api_hmac.new::0",
                "relevance_score": 3,
                "evidence_text": "hmac"
            }
        ]
    },
    {
        "query_id": "Q160",
        "question": "Why is collections.deque significantly faster than list for left-side insertions and deletions?",
        "category": "multi_document",
        "expected_answer": "collections.deque is implemented as a doubly linked list of blocks, providing O(1) time complexity for popleft() and appendleft(). A standard Python list is a contiguous array, requiring O(n) element shifting for pop(0) and insert(0, item).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_collections::api_collections.deque::0",
                "relevance_score": 3,
                "evidence_text": "deque"
            },
            {
                "chunk_id": "tutorial_datastructures::sec_using-lists-as-queues::0",
                "relevance_score": 3,
                "evidence_text": "deque"
            }
        ]
    },
    {
        "query_id": "Q161",
        "question": "How does data access differ between csv.reader and csv.DictReader?",
        "category": "multi_document",
        "expected_answer": "csv.reader parses each row into a list of strings accessed by integer index. csv.DictReader maps each row into an OrderedDict or dict with keys derived from the header line or fieldnames.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_csv::api_csv.reader::0",
                "relevance_score": 3,
                "evidence_text": "reader"
            },
            {
                "chunk_id": "library_csv::api_csv.DictReader::0",
                "relevance_score": 3,
                "evidence_text": "DictReader"
            }
        ]
    },
    {
        "query_id": "Q162",
        "question": "What is the key difference between tempfile.TemporaryFile and tempfile.NamedTemporaryFile?",
        "category": "multi_document",
        "expected_answer": "tempfile.TemporaryFile creates an anonymous file that has no directory entry on platforms that support it (unlinked immediately). tempfile.NamedTemporaryFile creates a file with a visible filesystem name accessible via .name.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_tempfile::api_tempfile.TemporaryFile::0",
                "relevance_score": 3,
                "evidence_text": "TemporaryFile"
            },
            {
                "chunk_id": "library_tempfile::api_tempfile.NamedTemporaryFile::0",
                "relevance_score": 3,
                "evidence_text": "NamedTemporaryFile"
            }
        ]
    },
    {
        "query_id": "Q163",
        "question": "Why is the logging module preferred over print() statements in production Python applications?",
        "category": "multi_document",
        "expected_answer": "logging provides configurable severity levels, module filtering, automatic formatting (timestamps, process IDs), multiple destinations (files, network sockets), and runtime verbosity control without changing code.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_logging::api_logging.basicConfig::0",
                "relevance_score": 3,
                "evidence_text": "basicConfig"
            },
            {
                "chunk_id": "library_functions::api_print::0",
                "relevance_score": 2,
                "evidence_text": "print"
            }
        ]
    },
    {
        "query_id": "Q164",
        "question": "How does array.array differ from a standard Python list in terms of homogeneity and memory efficiency?",
        "category": "multi_document",
        "expected_answer": "array.array stores contiguous uniform machine values of a single basic C type (like integers or floats), minimizing memory footprint. A Python list stores pointers to heterogeneous Python objects with substantial per-element pointer overhead.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_array::api_array.array::0",
                "relevance_score": 3,
                "evidence_text": "array"
            },
            {
                "chunk_id": "tutorial_datastructures::sec_more-on-lists::0",
                "relevance_score": 2,
                "evidence_text": "list"
            }
        ]
    },
    {
        "query_id": "Q165",
        "question": "When should you choose ThreadPoolExecutor over ProcessPoolExecutor in concurrent.futures?",
        "category": "multi_document",
        "expected_answer": "ThreadPoolExecutor is suited for I/O-bound tasks where worker threads spend most time waiting on sockets or disk. ProcessPoolExecutor bypasses the GIL for CPU-bound computations by executing jobs across worker processes.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_concurrent.futures::api_concurrent.futures.ThreadPoolExecutor::0",
                "relevance_score": 3,
                "evidence_text": "ThreadPoolExecutor"
            },
            {
                "chunk_id": "library_concurrent.futures::api_concurrent.futures.ProcessPoolExecutor::0",
                "relevance_score": 3,
                "evidence_text": "ProcessPoolExecutor"
            }
        ]
    },
    {
        "query_id": "Q166",
        "question": "How does math.sqrt differ from cmath.sqrt when passed a negative number?",
        "category": "multi_document",
        "expected_answer": "math.sqrt(-1) raises a ValueError because the math module is strictly for real-valued arithmetic. cmath.sqrt(-1) returns a complex number (1j) because cmath handles complex arithmetic.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_math::api_math.sqrt::0",
                "relevance_score": 3,
                "evidence_text": "sqrt"
            },
            {
                "chunk_id": "library_cmath::api_cmath.sqrt::0",
                "relevance_score": 3,
                "evidence_text": "sqrt"
            }
        ]
    },
    {
        "query_id": "Q167",
        "question": "What is the difference between blocking BSD sockets and asyncio streams for network communication?",
        "category": "multi_document",
        "expected_answer": "Standard socket operations block the calling OS thread until data arrives or timeouts expire. asyncio.open_connection provides StreamReader and StreamWriter instances that suspend coroutines cooperatively using non-blocking I/O.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_sockets::sec_creating-a-socket::1",
                "relevance_score": 3,
                "evidence_text": "socket"
            },
            {
                "chunk_id": "library_asyncio-stream::api_asyncio.open_connection::0",
                "relevance_score": 3,
                "evidence_text": "open_connection"
            }
        ]
    },
    {
        "query_id": "Q168",
        "question": "How does contextlib.closing differ from contextlib.suppress in context manager utility?",
        "category": "multi_document",
        "expected_answer": "contextlib.closing wraps an object with a .close() method ensuring .close() is called upon exit. contextlib.suppress(*exceptions) suppresses specified exception types raised within the context body without re-raising.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_contextlib::api_contextlib.closing::0",
                "relevance_score": 3,
                "evidence_text": "closing"
            },
            {
                "chunk_id": "library_contextlib::api_contextlib.suppress::0",
                "relevance_score": 3,
                "evidence_text": "suppress"
            }
        ]
    },
    {
        "query_id": "Q169",
        "question": "What is the difference between io.StringIO and io.BytesIO in in-memory stream processing?",
        "category": "multi_document",
        "expected_answer": "io.StringIO operates as an in-memory stream for unicode text strings (str), while io.BytesIO operates as an in-memory stream for binary data (bytes).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_io::api_io.StringIO::0",
                "relevance_score": 3,
                "evidence_text": "StringIO"
            },
            {
                "chunk_id": "library_io::api_io.BytesIO::0",
                "relevance_score": 3,
                "evidence_text": "BytesIO"
            }
        ]
    },
    {
        "query_id": "Q170",
        "question": "How do unittest and doctest differ in testing philosophy and execution style?",
        "category": "multi_document",
        "expected_answer": "unittest is a full xUnit-style test automation framework with test cases, suites, assertions, and test fixtures. doctest searches docstrings for interactive Python session sessions (>>>) to verify code examples match documented output.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_unittest::api_unittest.TestCase::0",
                "relevance_score": 3,
                "evidence_text": "TestCase"
            },
            {
                "chunk_id": "library_doctest::api_doctest.testmod::0",
                "relevance_score": 3,
                "evidence_text": "testmod"
            }
        ]
    },
    {
        "query_id": "Q171",
        "question": "What is the difference between copy.copy() and copy.deepcopy() on compound objects?",
        "category": "multi_document",
        "expected_answer": "copy.copy(x) creates a shallow copy, constructing a new compound object and inserting references to the original child objects. copy.deepcopy(x) recursively copies all child objects found within the original.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_copy::api_copy.copy::0",
                "relevance_score": 3,
                "evidence_text": "shallow copy"
            },
            {
                "chunk_id": "library_copy::api_copy.deepcopy::0",
                "relevance_score": 3,
                "evidence_text": "deep copy"
            }
        ]
    },
    {
        "query_id": "Q172",
        "question": "Why is frozenset suitable as a dictionary key or set element while set is not?",
        "category": "multi_document",
        "expected_answer": "set is mutable and therefore unhashable (lacks __hash__), preventing it from being used as a dictionary key or placed inside another set. frozenset is immutable and hashable, making it valid for keys and set membership.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functions::api_class set::0",
                "relevance_score": 3,
                "evidence_text": "set"
            },
            {
                "chunk_id": "library_functions::api_class frozenset::0",
                "relevance_score": 3,
                "evidence_text": "frozenset"
            }
        ]
    },
    {
        "query_id": "Q173",
        "question": "How does xml.etree.ElementTree differ from html.parser in parsing target and error tolerance?",
        "category": "multi_document",
        "expected_answer": "ElementTree is a tree-based parser designed for well-formed hierarchical XML documents. html.parser is a streaming event-driven parser designed to handle lenient, non-strict, and potentially malformed HTML formatting.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_xml.etree.elementtree::api_xml.etree.ElementTree.parse::0",
                "relevance_score": 3,
                "evidence_text": "ElementTree"
            },
            {
                "chunk_id": "library_html.parser::api_html.parser.HTMLParser::0",
                "relevance_score": 3,
                "evidence_text": "HTMLParser"
            }
        ]
    },
    {
        "query_id": "Q174",
        "question": "Why should time.monotonic() be used for measuring elapsed intervals instead of time.time()?",
        "category": "multi_document",
        "expected_answer": "time.time() returns system wall-clock time which can jump forward or backward due to NTP synchronization or daylight saving adjustments. time.monotonic() is a non-decreasing clock immune to system clock adjustments.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_time::api_time.monotonic::0",
                "relevance_score": 3,
                "evidence_text": "monotonic"
            },
            {
                "chunk_id": "library_time::api_time.time::0",
                "relevance_score": 2,
                "evidence_text": "time"
            }
        ]
    }
]

