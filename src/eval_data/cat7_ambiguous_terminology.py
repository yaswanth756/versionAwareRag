"""
Category 7: Ambiguous or Terminology-Based Questions (26 Questions)
Focus: Disambiguating Python terminology, nuanced language keywords, subtle behavioral distinctions, and nomenclature.
"""

QUESTIONS = [
    {
        "query_id": "Q175",
        "question": "What is the difference between a coroutine and a coroutine function in Python?",
        "category": "ambiguous_terminology",
        "expected_answer": "A coroutine function is the function definition declared with async def. A coroutine is the actual generator-like object returned when a coroutine function is called, which can then be awaited or scheduled as a task.",
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
        "query_id": "Q176",
        "question": "What is the difference between an Iterable and an Iterator in Python?",
        "category": "ambiguous_terminology",
        "expected_answer": "An Iterable is an object capable of returning its members one at a time by defining __iter__() or __getitem__(). An Iterator is the stateful stream object returned by iter(iterable) that implements __next__() and raises StopIteration when depleted.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_collections.abc::api_collections.abc.Iterable::0",
                "relevance_score": 3,
                "evidence_text": "ABC for classes that provide the __iter__() method."
            }
        ]
    },
    {
        "query_id": "Q177",
        "question": "What is the formal distinction between a parameter and an argument in Python?",
        "category": "ambiguous_terminology",
        "expected_answer": "A parameter is the named variable defined in the function signature header. An argument is the actual value or expression passed into the function when invoking it.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_controlflow::sec_special-parameters::0",
                "relevance_score": 3,
                "evidence_text": "positional-only"
            }
        ]
    },
    {
        "query_id": "Q178",
        "question": "What does the term 'dunder' mean in Python documentation and why is it used?",
        "category": "ambiguous_terminology",
        "expected_answer": "'Dunder' is Python shorthand for 'double underscore' (e.g. __init__ is pronounced 'dunder init'). It refers to special reserved method and attribute names defined by the Python language model.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::sec_id3::0",
                "relevance_score": 3,
                "evidence_text": "Special method names"
            }
        ]
    },
    {
        "query_id": "Q179",
        "question": "How do positional-only and keyword-only parameter delimiters (/ and *) function in function definitions?",
        "category": "ambiguous_terminology",
        "expected_answer": "Parameters before a forward slash (/) are positional-only and cannot be passed by keyword. Parameters after an asterisk (*) are keyword-only and must be passed with their parameter names.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_controlflow::sec_special-parameters::0",
                "relevance_score": 3,
                "evidence_text": "keyword-only"
            }
        ]
    },
    {
        "query_id": "Q180",
        "question": "What is a data descriptor and how does its precedence differ from non-data descriptors?",
        "category": "ambiguous_terminology",
        "expected_answer": "A data descriptor defines __set__() or __delete__() (or both). It takes precedence over instance dictionary lookup. A non-data descriptor only defines __get__() and is overridden by an entry in the instance __dict__.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__set__::0",
                "relevance_score": 3,
                "evidence_text": "data descriptor"
            }
        ]
    },
    {
        "query_id": "Q181",
        "question": "What is the difference between a module and a package in Python's import system?",
        "category": "ambiguous_terminology",
        "expected_answer": "A module is a single Python file (.py) containing definitions. A package is a directory containing modules and an __init__.py file (or a namespace package spanning multiple directories) allowing dotted module navigation.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_modules::sec_the-module-search-path::0",
                "relevance_score": 3,
                "evidence_text": "sys.path"
            }
        ]
    },
    {
        "query_id": "Q182",
        "question": "What is the difference between str() (__str__) and repr() (__repr__)?",
        "category": "ambiguous_terminology",
        "expected_answer": "repr() computes an official string representation containing unambiguous information suitable for debugging and reconstruction. str() computes an informal, human-readable display string.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functions::api_repr::0",
                "relevance_score": 3,
                "evidence_text": "Return a string containing a printable representation of an object."
            }
        ]
    },
    {
        "query_id": "Q183",
        "question": "What is the distinction between a generator function and a generator iterator?",
        "category": "ambiguous_terminology",
        "expected_answer": "A generator function is the function containing a yield expression. A generator iterator is the object created when calling that function, producing yielded values via next().",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_expressions::api_generator.__next__::0",
                "relevance_score": 3,
                "evidence_text": "generator"
            }
        ]
    },
    {
        "query_id": "Q184",
        "question": "How does @classmethod differ from @staticmethod in Python class definitions?",
        "category": "ambiguous_terminology",
        "expected_answer": "A classmethod receives the class object (cls) implicitly as its first argument, allowing access to class state and factory methods. A staticmethod receives neither the instance nor the class, behaving like a plain function scoped inside the class.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functions::api_classmethod::0",
                "relevance_score": 3,
                "evidence_text": "Transform a method into a class method."
            }
        ]
    },
    {
        "query_id": "Q185",
        "question": "What is a closure and what are free variables in Python scope execution?",
        "category": "ambiguous_terminology",
        "expected_answer": "A closure is an inner function that retains references to variables from its enclosing lexical scope ('free variables') even after the enclosing scope has finished executing.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_executionmodel::sec_interaction-with-dynamic-features::0",
                "relevance_score": 3,
                "evidence_text": "free variables"
            }
        ]
    },
    {
        "query_id": "Q186",
        "question": "What is the difference between nominal subtyping with ABCs and structural subtyping with Protocols?",
        "category": "ambiguous_terminology",
        "expected_answer": "Abstract Base Classes (ABCs) use nominal subtyping where a class must explicitly inherit from the ABC (or register with it). typing.Protocol uses structural subtyping ('static duck typing') where any class implementing the matching methods satisfies the protocol without explicit inheritance.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_typing::api_typing.Protocol::0",
                "relevance_score": 3,
                "evidence_text": "Base class for protocol classes."
            }
        ]
    },
    {
        "query_id": "Q187",
        "question": "How do str and bytes differ in Python 3 data representation?",
        "category": "ambiguous_terminology",
        "expected_answer": "str represents text as an immutable sequence of abstract Unicode characters (code points). bytes represents raw binary data as an immutable sequence of 8-bit integers (0 <= x < 256).",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_functions::api_class bytes::0",
                "relevance_score": 3,
                "evidence_text": "immutable sequence of integers in the range 0 <= x < 256"
            }
        ]
    },
    {
        "query_id": "Q188",
        "question": "What makes an object 'hashable' in Python and why must hash values remain immutable?",
        "category": "ambiguous_terminology",
        "expected_answer": "An object is hashable if it has a hash value that never changes during its lifetime (implements __hash__()) and can be compared to other objects (implements __eq__()). Hash values must remain immutable so hash tables (dicts/sets) can locate buckets reliably.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_collections.abc::api_collections.abc.Hashable::0",
                "relevance_score": 3,
                "evidence_text": "ABC for classes that provide the __hash__() method."
            }
        ]
    },
    {
        "query_id": "Q189",
        "question": "How do dictionary view objects differ from regular lists in Python?",
        "category": "ambiguous_terminology",
        "expected_answer": "Dictionary view objects (dict.keys(), dict.values(), dict.items()) are dynamic windows into the dictionary's contents. If the dictionary is modified after the view is created, the view reflects those changes immediately without copying data.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_stdtypes::sec_dictionary-view-objects::0",
                "relevance_score": 3,
                "evidence_text": "Dictionary view objects"
            }
        ]
    },
    {
        "query_id": "Q190",
        "question": "What is the difference between a compile-time syntax error and a runtime exception in Python?",
        "category": "ambiguous_terminology",
        "expected_answer": "A syntax error (parsing error) occurs while the Python compiler parses source code before execution begins. A runtime exception occurs during program execution when valid code encounters an error condition.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_errors::sec_syntax-errors::0",
                "relevance_score": 3,
                "evidence_text": "Syntax errors, also known as parsing errors"
            }
        ]
    },
    {
        "query_id": "Q191",
        "question": "What is the difference between the pass statement and the Ellipsis (...) singleton?",
        "category": "ambiguous_terminology",
        "expected_answer": "pass is a statement that executes as a no-op placeholder. Ellipsis (...) is an actual object singleton of type types.EllipsisType, commonly used in type annotations, multidimensional numpy slicing, and stub functions.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_constants::api_Ellipsis::0",
                "relevance_score": 3,
                "evidence_text": "The same as the ellipsis literal “...”."
            }
        ]
    },
    {
        "query_id": "Q192",
        "question": "What is the difference between the global namespace and the builtins namespace?",
        "category": "ambiguous_terminology",
        "expected_answer": "The global namespace contains variables and functions defined at the module level. The builtins namespace contains core Python built-in functions, exceptions, and types (like len, print, ValueError) searched as a fallback when names are not found globally.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_executionmodel::sec_resolution-of-names::0",
                "relevance_score": 3,
                "evidence_text": "global namespace"
            }
        ]
    },
    {
        "query_id": "Q193",
        "question": "How does short-circuit evaluation work for Python boolean operators 'and' and 'or'?",
        "category": "ambiguous_terminology",
        "expected_answer": "Boolean operators 'and' and 'or' short-circuit: 'x and y' evaluates x first, returning x if false without evaluating y. 'x or y' returns x if true without evaluating y. They return the operand itself rather than a coerced boolean.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_expressions::sec_boolean-operations::0",
                "relevance_score": 3,
                "evidence_text": "The expression x and y first evaluates x; if x is false, its value is returned"
            }
        ]
    },
    {
        "query_id": "Q194",
        "question": "What determines whether a non-boolean Python object is considered truthy or falsy?",
        "category": "ambiguous_terminology",
        "expected_answer": "An object's truth value is determined by __bool__(). If undefined, Python calls __len__(). Empty collections (len == 0), numeric zeros (0, 0.0), and None evaluate to falsy; non-empty containers and non-zero numbers evaluate to truthy.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_datamodel::api_object.__bool__::0",
                "relevance_score": 3,
                "evidence_text": "truth value testing"
            }
        ]
    },
    {
        "query_id": "Q195",
        "question": "How does os.environ differ from os.getenv() for environment variable retrieval?",
        "category": "ambiguous_terminology",
        "expected_answer": "os.environ is a mutable mapping where missing keys raise KeyError when indexed. os.getenv(key, default=None) is a helper function that returns None or a default value when the key is absent.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_os::api_os.environ::0",
                "relevance_score": 3,
                "evidence_text": "A mapping object where keys and values are strings that represent the process environment."
            }
        ]
    },
    {
        "query_id": "Q196",
        "question": "Why does 'a == b' evaluate to True while 'a is b' evaluates to False for independently constructed identical objects?",
        "category": "ambiguous_terminology",
        "expected_answer": "'a == b' evaluates value equality via __eq__(), which checks if contents match. 'a is b' tests identity, returning True only if both variables point to the exact same memory address.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_expressions::sec_comparisons::0",
                "relevance_score": 3,
                "evidence_text": "Comparisons"
            }
        ]
    },
    {
        "query_id": "Q197",
        "question": "What is the functional difference between shallow copy and deep copy for nested lists?",
        "category": "ambiguous_terminology",
        "expected_answer": "A shallow copy duplicates the outer list container but copies references to the inner nested lists. Modifying an inner list affects both copies. A deep copy recursively duplicates all nested objects.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_copy::api_copy.copy::0",
                "relevance_score": 3,
                "evidence_text": "shallow copy"
            }
        ]
    },
    {
        "query_id": "Q198",
        "question": "What is the purpose of the nonlocal statement in nested Python functions?",
        "category": "ambiguous_terminology",
        "expected_answer": "The nonlocal statement causes listed identifiers to refer to previously bound variables in the nearest enclosing function scope (excluding globals), allowing nested closures to rebind enclosing variables.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "reference_simple_stmts::sec_the-nonlocal-statement::0",
                "relevance_score": 3,
                "evidence_text": "nonlocal"
            }
        ]
    },
    {
        "query_id": "Q199",
        "question": "What is the fundamental difference between synchronous blocking I/O and asynchronous event loops?",
        "category": "ambiguous_terminology",
        "expected_answer": "Synchronous I/O blocks the thread of execution until the operation completes. Asynchronous event loops register I/O operations with OS polling mechanisms (like epoll or kqueue) and switch to running other tasks while waiting.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "library_asyncio::sec_module-asyncio::0",
                "relevance_score": 3,
                "evidence_text": "asyncio is a library to write concurrent code using the async/await syntax."
            }
        ]
    },
    {
        "query_id": "Q200",
        "question": "Why does using a mutable default argument like def f(x=[]) cause shared state across calls?",
        "category": "ambiguous_terminology",
        "expected_answer": "In Python, default parameter expressions are evaluated once when the function definition is executed, not each time the function is called. Mutating a default argument modifies the shared object across all subsequent calls.",
        "needs_review": False,
        "relevant_chunks": [
            {
                "chunk_id": "tutorial_controlflow::sec_default-argument-values::0",
                "relevance_score": 3,
                "evidence_text": "default value is evaluated only once"
            }
        ]
    }
]

