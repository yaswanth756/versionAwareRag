"""
Category 2: Conceptual Questions (30 Questions)
Focus: Python data model, execution model, scoping rules, protocols, memory management, and runtime design.
"""

QUESTIONS = [
    {
        "query_id": "Q031",
        "question": "What is the purpose of the __get__ method in the Python descriptor protocol?",
        "category": "conceptual",
        "expected_answer": "In Python's descriptor protocol, __get__(self, instance, owner=None) is called to get the attribute of the owner class (when accessed via class) or of an instance of that class (when accessed via instance).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__get__::0",
                "relevance_score": 3,
                "evidence_text": "Called to get the attribute of the owner class"
            }
        ]
    },
    {
        "query_id": "Q032",
        "question": "What distinguishes a data descriptor from a non-data descriptor in Python?",
        "category": "conceptual",
        "expected_answer": "A descriptor that defines __set__() or __delete__() is considered a 'data descriptor'. Data descriptors take precedence over an instance dictionary during attribute lookup, whereas non-data descriptors (which only define __get__(), like normal methods) can be overridden by instance dictionary entries.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__set__::0",
                "relevance_score": 3,
                "evidence_text": "Note, adding __set__() or __delete__() changes the kind of descriptor to a “data descriptor”."
            }
        ]
    },
    {
        "query_id": "Q033",
        "question": "What algorithm does Python use to compute the Method Resolution Order (MRO) for multiple inheritance?",
        "category": "conceptual",
        "expected_answer": "Python uses the C3 Method Resolution Order (MRO) algorithm, introduced in Python 2.3 and continued in Python 3, ensuring monotonicity and respecting local precedence order across complex inheritance graphs.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_mro::sec_the-python-2-3-method-resolution-order::0",
                "relevance_score": 3,
                "evidence_text": "This document is intended for Python programmers who want to understand the C3 Method Resolution Order"
            }
        ]
    },
    {
        "query_id": "Q034",
        "question": "How does __init_subclass__ serve as a lightweight alternative to custom metaclasses?",
        "category": "conceptual",
        "expected_answer": "__init_subclass__ is called whenever the containing class is subclassed. It receives the new subclass as its first argument and can customize class creation without requiring the complexity of defining a custom metaclass.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__init_subclass__::0",
                "relevance_score": 3,
                "evidence_text": "This method is called whenever the containing class is subclassed."
            }
        ]
    },
    {
        "query_id": "Q035",
        "question": "What role does __enter__ play in the Python context management protocol?",
        "category": "conceptual",
        "expected_answer": "In Python's context management protocol, __enter__() enters the runtime context and optionally returns an object that is bound to the target specified in the with statement's as clause.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__enter__::0",
                "relevance_score": 3,
                "evidence_text": "Enter the runtime context related to this object."
            }
        ]
    },
    {
        "query_id": "Q036",
        "question": "How does defining __slots__ in a class affect memory allocation and instance attributes?",
        "category": "conceptual",
        "expected_answer": "__slots__ reserves space for a fixed set of attributes and suppresses the creation of an instance __dict__ (unless explicitly included), reducing memory consumption and preventing arbitrary attribute assignment on instances.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::sec_slots::0",
                "relevance_score": 3,
                "evidence_text": "__slots__"
            }
        ]
    },
    {
        "query_id": "Q037",
        "question": "How does Python's structural pattern matching evaluate compound match statements?",
        "category": "conceptual",
        "expected_answer": "In Python compound statements, a match statement takes an expression and compares its value against successive patterns in case blocks, executing the block belonging to the first matching pattern.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_compound_stmts::sec_compound-statements::1",
                "relevance_score": 3,
                "evidence_text": "compound_stmt"
            }
        ]
    },
    {
        "query_id": "Q038",
        "question": "What is the semantic purpose of a yield expression in Python functions?",
        "category": "conceptual",
        "expected_answer": "A yield expression turns a function into a generator function. When called, it returns a generator iterator that yields values one at a time, suspending local execution state between yields.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_expressions::sec_yield-expressions::0",
                "relevance_score": 3,
                "evidence_text": "The yield expression is used when defining a generator function"
            }
        ]
    },
    {
        "query_id": "Q039",
        "question": "How does Python resolve variable names across nested scopes?",
        "category": "conceptual",
        "expected_answer": "A scope defines the visibility of a name within a block. When a name is referenced in a code block, Python resolves it using the nearest enclosing scope (the LEGB rule: Local, Enclosing, Global, Builtins). If not found, NameError is raised.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_executionmodel::sec_resolution-of-names::0",
                "relevance_score": 3,
                "evidence_text": "A scope defines the visibility of a name within a block."
            }
        ]
    },
    {
        "query_id": "Q040",
        "question": "What is PEP 684 per-interpreter GIL and how does it affect multi-core parallelism?",
        "category": "conceptual",
        "expected_answer": "PEP 684 introduces a per-interpreter Global Interpreter Lock (GIL) in Python 3.12, allowing sub-interpreters to run concurrently with their own independent GIL, enabling true parallel Python code execution across multiple CPU cores.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "whatsnew_3.12::sec_pep-684-a-per-interpreter-gil::0",
                "relevance_score": 3,
                "evidence_text": "PEP 684 introduces a per-interpreter GIL"
            }
        ]
    },
    {
        "query_id": "Q041",
        "question": "What is an asyncio Task and how does it relate to coroutines?",
        "category": "conceptual",
        "expected_answer": "An asyncio Task wraps a coroutine into a Future-like object and schedules its execution on the event loop. Tasks allow concurrent execution of multiple coroutines.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_asyncio-task::api_asyncio.create_task::0",
                "relevance_score": 3,
                "evidence_text": "Wrap the coro coroutine into a Task and schedule its execution."
            }
        ]
    },
    {
        "query_id": "Q042",
        "question": "How does the try statement handle compound exception handling with except*?",
        "category": "conceptual",
        "expected_answer": "In Python 3.11+, try statements can use except* clauses to match and handle subsets of exceptions contained within an ExceptionGroup, allowing multiple handlers to execute for concurrent failures.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_compound_stmts::sec_the-try-statement::0",
                "relevance_score": 3,
                "evidence_text": "try_stmt"
            }
        ]
    },
    {
        "query_id": "Q043",
        "question": "What are rich comparison methods in Python and how do they correspond to comparison operators?",
        "category": "conceptual",
        "expected_answer": "Rich comparison methods (__lt__, __le__, __eq__, __ne__, __gt__, __ge__) overload comparison operators (<, <=, ==, !=, >, >=). They can return NotImplemented if the comparison between operands is unsupported.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__lt__::0",
                "relevance_score": 3,
                "evidence_text": "These are the so-called “rich comparison” methods."
            }
        ]
    },
    {
        "query_id": "Q044",
        "question": "How does Python evaluate an object's truth value with __bool__ and __len__?",
        "category": "conceptual",
        "expected_answer": "Python calls __bool__() to determine an object's truth value. If __bool__() is not defined, Python falls back to __len__(), considering the object true if its length is non-zero, and true by default if neither is defined.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__bool__::0",
                "relevance_score": 3,
                "evidence_text": "Called to implement truth value testing and the built-in operation bool()"
            }
        ]
    },
    {
        "query_id": "Q045",
        "question": "How does CPython manage memory for cyclic references that reference counting cannot reclaim?",
        "category": "conceptual",
        "expected_answer": "CPython uses a cyclic generational garbage collector (accessible via the gc module) that identifies and reclaims unreachable reference cycles that reference counting alone cannot collect.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_gc::api_gc.collect::0",
                "relevance_score": 3,
                "evidence_text": "With no arguments, run a full collection."
            }
        ]
    },
    {
        "query_id": "Q046",
        "question": "What is the buffer protocol and how did PEP 688 expose it in Python 3.12?",
        "category": "conceptual",
        "expected_answer": "PEP 688 allows Python code to directly access the buffer protocol using collections.abc.Buffer, enabling type annotations for objects that provide raw binary buffers (such as bytes, bytearray, and memoryview).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "whatsnew_3.12::sec_pep-688-making-the-buffer-protocol-accessible-in-python::0",
                "relevance_score": 3,
                "evidence_text": "PEP 688: Making the buffer protocol accessible in Python"
            }
        ]
    },
    {
        "query_id": "Q047",
        "question": "What is the purpose of the abc.abstractmethod decorator?",
        "category": "conceptual",
        "expected_answer": "abc.abstractmethod defines abstract methods on Abstract Base Classes (ABCs). A class containing abstract methods cannot be instantiated unless all abstract methods are overridden in a concrete subclass.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_abc::api_abc.abstractmethod::0",
                "relevance_score": 3,
                "evidence_text": "A decorator indicating abstract methods."
            }
        ]
    },
    {
        "query_id": "Q048",
        "question": "How does implementing __call__ make an instance callable in Python?",
        "category": "conceptual",
        "expected_answer": "Implementing __call__(self, [args...]) on a class enables its instances to be invoked like functions using parentheses, turning them into callable objects.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__call__::0",
                "relevance_score": 3,
                "evidence_text": "Called when the instance is “called” as a function"
            }
        ]
    },
    {
        "query_id": "Q049",
        "question": "What defines an iterator in Python collections.abc?",
        "category": "conceptual",
        "expected_answer": "In collections.abc, an Iterator is an Iterable that implements __next__() (which returns items one by one and raises StopIteration when exhausted) and an __iter__() method that returns self.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_collections.abc::api_collections.abc.Iterator::0",
                "relevance_score": 3,
                "evidence_text": "Iterator"
            }
        ]
    },
    {
        "query_id": "Q050",
        "question": "What is a weak reference in Python and why is it used?",
        "category": "conceptual",
        "expected_answer": "A weak reference (weakref.ref) allows referencing an object without increasing its reference count. When the only remaining references to an object are weak, the object is eligible for garbage collection, preventing memory leaks in caches and circular graphs.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_weakref::api_weakref.ref::0",
                "relevance_score": 3,
                "evidence_text": "Return a weak reference to object."
            }
        ]
    },
    {
        "query_id": "Q051",
        "question": "What is the recommended best practice for inspecting an object's annotations in Python 3.10 and newer?",
        "category": "conceptual",
        "expected_answer": "In Python 3.10+, inspect.get_annotations() is the recommended standard tool to access an object's annotations dict, safely handling stringized annotations, classes, modules, and callables.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "howto_annotations::sec_annotations-best-practices::0",
                "relevance_score": 3,
                "evidence_text": "best practices for working with annotations dicts"
            }
        ]
    },
    {
        "query_id": "Q052",
        "question": "What are Python bytecode instructions and how does the dis module interact with them?",
        "category": "conceptual",
        "expected_answer": "The Python compiler translates source code into bytecode instructions executed by the CPython virtual machine. The dis module disassembles code objects into human-readable bytecode instructions and opcodes.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_dis::sec_python-bytecode-instructions::0",
                "relevance_score": 3,
                "evidence_text": "The Python compiler currently generates the following bytecode instructions."
            }
        ]
    },
    {
        "query_id": "Q053",
        "question": "How does eval() differ from exec() in terms of return values and supported constructs?",
        "category": "conceptual",
        "expected_answer": "eval() takes an expression string and evaluates it, returning the computed expression value. In contrast, exec() dynamically executes statements or code suites and returns None.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functions::api_eval::0",
                "relevance_score": 3,
                "evidence_text": "The expression argument is parsed and evaluated as a Python expression"
            }
        ]
    },
    {
        "query_id": "Q054",
        "question": "What role does sys.meta_path play in Python's import machinery?",
        "category": "conceptual",
        "expected_answer": "sys.meta_path is a list of meta path finder objects that Python queries during import processing to find and load modules. Custom finders can be registered on sys.meta_path to customize module loading.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_import::sec_the-import-system::0",
                "relevance_score": 3,
                "evidence_text": "There is no longer any implicit import machinery - the full import system is exposed through sys.meta_path."
            }
        ]
    },
    {
        "query_id": "Q055",
        "question": "How does PEP 695 type parameter syntax simplify generic classes and functions in Python 3.12?",
        "category": "conceptual",
        "expected_answer": "PEP 695 introduces compact generic syntax (e.g., class Stack[T]: and def func[T](x: T) -> T:) and the type alias statement (type Point = tuple[float, float]), eliminating verbose TypeVar and Generic boilerplate.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "whatsnew_3.12::sec_pep-695-type-parameter-syntax::0",
                "relevance_score": 3,
                "evidence_text": "PEP 695: Type Parameter Syntax"
            }
        ]
    },
    {
        "query_id": "Q056",
        "question": "What lifecycle states can a coroutine object exist in according to inspect.getcoroutinestate?",
        "category": "conceptual",
        "expected_answer": "inspect.getcoroutinestate() can return one of four states: CORO_CREATED (waiting to start), CORO_RUNNING (currently executing), CORO_SUSPENDED (suspended at an await expression), or CORO_CLOSED (execution finished).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_inspect::api_inspect.getcoroutinestate::0",
                "relevance_score": 3,
                "evidence_text": "Get current state of a coroutine object."
            }
        ]
    },
    {
        "query_id": "Q057",
        "question": "How does super() determine the next class to delegate to in multiple inheritance?",
        "category": "conceptual",
        "expected_answer": "super() returns a proxy object that delegates method calls based on the Method Resolution Order (MRO) of the caller's class or instance, enabling cooperative multiple inheritance.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functions::api_super::0",
                "relevance_score": 3,
                "evidence_text": "Return a proxy object that delegates method calls to a parent or sibling class of type."
            }
        ]
    },
    {
        "query_id": "Q058",
        "question": "What does threading.active_count() report about active threads in a Python process?",
        "category": "conceptual",
        "expected_answer": "threading.active_count() returns the number of Thread objects currently alive in the Python process, matching the length of the list returned by threading.enumerate().",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_threading::api_threading.active_count::0",
                "relevance_score": 3,
                "evidence_text": "Return the number of Thread objects currently alive."
            }
        ]
    },
    {
        "query_id": "Q059",
        "question": "How does __init__.py establish a directory as a regular Python package?",
        "category": "conceptual",
        "expected_answer": "The presence of an __init__.py file informs Python that a directory is a regular package, preventing directories with common names from shadowing modules, and allowing initialization code or __all__ export lists to be defined.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_modules::sec_packages::0",
                "relevance_score": 3,
                "evidence_text": "The __init__.py files are required to make Python treat directories containing the file as packages"
            }
        ]
    },
    {
        "query_id": "Q060",
        "question": "What is PEP 709 comprehension inlining in Python 3.12 and how does it improve performance?",
        "category": "conceptual",
        "expected_answer": "PEP 709 inlines list, dict, and set comprehensions into the containing code block rather than creating a temporary single-use function object, resulting in faster execution and reduced call overhead.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "whatsnew_3.12::sec_pep-709-comprehension-inlining::0",
                "relevance_score": 3,
                "evidence_text": "PEP 709: Comprehension inlining"
            }
        ]
    }
]

