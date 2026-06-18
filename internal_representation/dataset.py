"""
Step 1 — Dataset Preparation.

Generates a labeled dataset of synthetic conversations for expertise probing.
Add more templates to TEMPLATES to improve coverage (aim for ~30 per level).

To use your own labeled data instead:
    Pass --data-file /path/to/data.json to main.py.
    Expected schema:
        {
            "conversations": ["User: ...\nAssistant: ..."],
            "labels": {"expertise": ["novice", "expert", ...]}
        }
"""

import json
import os
import random
from typing import Dict, List, Tuple

from config import DataConfig


# ---------------------------------------------------------------------------
# Conversation templates keyed by expertise level.
# Each template is a list of (role, message) turns (aim for 8-10 turns).
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, List[List[Tuple[str, str]]]] = {

    # ── NOVICE ──────────────────────────────────────────────────────────────
    "novice": [
        [
            ("user", "Hi! I just started learning Python. What is a variable?"),
            ("assistant", "A variable is like a labeled box where you store data. You give it a name and put a value in it."),
            ("user", "Oh cool! So if I write x = 5, the box is called x and holds 5?"),
            ("assistant", "Exactly! And you can change the value inside anytime — just write x = 10 later."),
            ("user", "What if I want to store words instead of numbers?"),
            ("assistant", "Wrap the text in quotes, like name = 'Alice'. That's called a string."),
            ("user", "Can I put a number and a word together in one variable?"),
            ("assistant", "Not directly — they're different types. But you can convert them when needed."),
            ("user", "This is a lot to take in. Is there a simple rule to remember?"),
            ("assistant", "Just think: name on the left, value on the right, equals sign in the middle. That's all a variable is."),
        ],
        [
            ("user", "What's a for loop? I keep seeing it in examples but I don't get it."),
            ("assistant", "A for loop repeats a block of code for each item in a list or range."),
            ("user", "Can you show me a tiny example?"),
            ("assistant", "Sure: `for i in range(3): print(i)` prints 0, then 1, then 2."),
            ("user", "Why does it start at 0 and not 1?"),
            ("assistant", "Python counts from 0 by default. It's a convention in most programming languages."),
            ("user", "That's confusing. What if I want 1, 2, 3?"),
            ("assistant", "Use `range(1, 4)` — the first number is where it starts, the second is where it stops (not included)."),
            ("user", "Oh okay. What happens inside the loop? How does Python know what belongs to it?"),
            ("assistant", "Indentation — the lines inside the loop must be shifted in by 4 spaces. That tells Python they're part of the loop."),
        ],
        [
            ("user", "I keep getting a NameError. It says 'name x is not defined'. What does that mean?"),
            ("assistant", "It means you're trying to use a variable called x before you've created it."),
            ("user", "But I did write x somewhere in my code!"),
            ("assistant", "Check that you wrote it *before* the line that's failing. Python reads code top to bottom."),
            ("user", "Oh! I used it on line 3 but set it on line 7. Is that the problem?"),
            ("assistant", "Yes, exactly. Move the assignment above the usage and the error will go away."),
            ("user", "Okay I moved it. Now I get a different error — TypeError."),
            ("assistant", "TypeErrors usually mean you're mixing incompatible types, like trying to add a number to a string."),
            ("user", "How do I know what type a variable is?"),
            ("assistant", "Use `type(x)` — it tells you whether it's an int, str, list, etc."),
        ],
        [
            ("user", "What does it mean when my code has an IndentationError?"),
            ("assistant", "It means the spacing at the start of a line is wrong. Python uses indentation to group code."),
            ("user", "I don't understand what grouping means here."),
            ("assistant", "For example, code inside an if statement must be indented — shifted to the right — to show it belongs there."),
            ("user", "How many spaces should I use?"),
            ("assistant", "4 spaces is the standard in Python. The important thing is to be consistent — don't mix tabs and spaces."),
            ("user", "My editor uses tabs. Is that a problem?"),
            ("assistant", "It can be. Set your editor to convert tabs to 4 spaces automatically — most editors have this option."),
            ("user", "I changed the setting. My code runs now but the if statement doesn't seem to do anything."),
            ("assistant", "Check your condition — if the condition is never True, the code inside won't run. Try printing the value you're testing."),
        ],
        [
            ("user", "What is a function and why would I need one?"),
            ("assistant", "A function is a reusable block of code. You define it once and call it whenever you need it."),
            ("user", "How do I make one?"),
            ("assistant", "Use `def`, then a name, then parentheses: `def say_hello(): print('Hello!')`. Then call it with `say_hello()`."),
            ("user", "What are the parentheses for?"),
            ("assistant", "They hold parameters — inputs you can pass into the function. If there are none, you still need empty parentheses."),
            ("user", "Can a function give me back a result?"),
            ("assistant", "Yes — use `return`. Like `def add(a, b): return a + b`. Then `result = add(3, 4)` gives you 7."),
            ("user", "What happens if I forget the return?"),
            ("assistant", "The function returns None by default, which means nothing useful came back. That's a common bug when you expect a value."),
            ("user", "Oh, I think that's my bug! I was printing the result of a function that had no return statement."),
            ("assistant", "That's it. Add `return` before the value you want to send back and it'll work."),
        ],
        [
            ("user", "I want to read a file in Python. How do I do that?"),
            ("assistant", "Use `open('filename.txt')` to open the file, then `.read()` to get the contents."),
            ("user", "Do I need to close the file after?"),
            ("assistant", "Yes, always. But the easiest way is to use a `with` block — it closes the file automatically."),
            ("user", "Can you show me what that looks like?"),
            ("assistant", "`with open('file.txt') as f: contents = f.read()`. After the block, the file is closed."),
            ("user", "What if the file doesn't exist?"),
            ("assistant", "You'll get a FileNotFoundError. You can handle it with a try/except block."),
            ("user", "I haven't learned try/except yet. Is there a simpler way to check first?"),
            ("assistant", "Yes — `import os` and then `os.path.exists('file.txt')` returns True or False."),
            ("user", "That's much easier for now. I'll learn try/except later."),
            ("assistant", "Good plan. It's fine to learn things step by step — start with what makes sense to you."),
        ],
        [
            ("user", "What is a list and how is it different from just having separate variables?"),
            ("assistant", "A list holds multiple values in one place, in order. Instead of x=1, y=2, z=3 you have nums = [1, 2, 3]."),
            ("user", "How do I get one item out of the list?"),
            ("assistant", "Use an index in square brackets: `nums[0]` gives the first item. Remember, counting starts at 0."),
            ("user", "How do I add something to a list?"),
            ("assistant", "Use `.append()`: `nums.append(4)` adds 4 to the end of the list."),
            ("user", "And how do I remove something?"),
            ("assistant", "Use `.remove(value)` to remove by value, or `del nums[index]` to remove by position."),
            ("user", "What if I want to go through every item in the list?"),
            ("assistant", "Use a for loop: `for item in nums: print(item)`. It goes through each one automatically."),
            ("user", "This is starting to make sense. Is a list the main way to store multiple things?"),
            ("assistant", "It's the most common, yes. Later you'll learn about dictionaries and sets, but lists are a great starting point."),
        ],
        [
            ("user", "How do I get input from the user while my program is running?"),
            ("assistant", "Use the `input()` function. It pauses the program and waits for the user to type something."),
            ("user", "How do I store what they type?"),
            ("assistant", "Assign it to a variable: `name = input('What is your name? ')`."),
            ("user", "Can I use that input as a number?"),
            ("assistant", "Not directly — input always returns a string. Convert it with `int()` or `float()` first."),
            ("user", "What if the user types something that isn't a number and I try to convert it?"),
            ("assistant", "You'll get a ValueError. That's something to handle with try/except once you're ready for that."),
            ("user", "For now I'll just assume they enter numbers. My program asks for age."),
            ("assistant", "That's fine for practice. Just write `age = int(input('Enter your age: '))` and you're good."),
            ("user", "I tried it and it works! Thank you."),
            ("assistant", "Great work. The more small programs you build, the faster it clicks."),
        ],
        [
            ("user", "I installed a package with pip but Python says it can't find it. What's wrong?"),
            ("assistant", "You might have multiple Python installations. The pip you ran might belong to a different one than the Python you're using."),
            ("user", "How do I tell which one I'm using?"),
            ("assistant", "Run `python --version` and `pip --version` in your terminal. The paths should match."),
            ("user", "They show different paths. How do I fix it?"),
            ("assistant", "Use `python -m pip install packagename` — this installs directly into the Python you're currently running."),
            ("user", "That worked! It installed. But now I'm wondering — should I be using a virtual environment?"),
            ("assistant", "Yes, virtual environments keep each project's packages separate. Run `python -m venv venv` to create one."),
            ("user", "And how do I turn it on?"),
            ("assistant", "On Windows: `venv\\Scripts\\activate`. On Mac/Linux: `source venv/bin/activate`. You'll see the name in your terminal prompt."),
            ("user", "I activated it and installed the package again. Now it works properly!"),
            ("assistant", "Perfect. Get in the habit of activating the environment before working on any project."),
        ],
        [
            ("user", "What does `if __name__ == '__main__':` mean? I see it everywhere."),
            ("assistant", "It's a guard that runs code only when you execute the file directly, not when it's imported by another file."),
            ("user", "I don't understand the difference between running and importing."),
            ("assistant", "When you run `python myfile.py`, Python sets `__name__` to `'__main__'`. When another file imports it, `__name__` is the module name instead."),
            ("user", "So the code inside won't run if someone imports my file?"),
            ("assistant", "Exactly. It protects your main logic from running accidentally when someone imports your functions."),
            ("user", "Should I always add it to my files?"),
            ("assistant", "It's good practice when your file has a main script section. For files that are purely functions/classes, it's less necessary."),
            ("user", "My file has both functions and a script that calls them. Should I put the script part inside that block?"),
            ("assistant", "Yes — put the calling code inside `if __name__ == '__main__':`. Keep your functions outside so they can be imported freely."),
            ("user", "Got it. I'll reorganize my file that way."),
            ("assistant", "Good instinct. It makes your code more reusable from the start."),
        ],
    ],

    # ── INTERMEDIATE ────────────────────────────────────────────────────────
    "intermediate": [
        [
            ("user", "I'm trying to decide between using a class and a named tuple for my data objects."),
            ("assistant", "If you need mutability and methods, use a class. For immutable records, named tuples are leaner."),
            ("user", "I've been using dataclasses lately — they feel like a nice middle ground."),
            ("assistant", "They are. You get type hints, default values, and auto-generated __repr__ with minimal boilerplate."),
            ("user", "What about frozen dataclasses versus named tuples for performance?"),
            ("assistant", "Named tuples are slightly faster and more memory-efficient since they're built on tuples. Frozen dataclasses have a touch more overhead."),
            ("user", "For my use case the data is read-only config. Would you recommend one over the other?"),
            ("assistant", "Either works, but I'd lean toward frozen dataclasses — better IDE support and you can add validation methods later."),
            ("user", "Does `slots=True` on a dataclass close the performance gap?"),
            ("assistant", "Yes, Python 3.10+ `slots=True` reduces per-instance memory and speeds up attribute access significantly."),
            ("user", "I'll go with a frozen dataclass with slots then. Thanks."),
            ("assistant", "Good call. You get the clarity of dataclasses and near-tuple performance."),
        ],
        [
            ("user", "I'm having trouble with async/await. My coroutines are running sequentially even though I expected them to run concurrently."),
            ("assistant", "Are you awaiting them one by one, or using asyncio.gather?"),
            ("user", "One by one — `await fetch(a)` then `await fetch(b)`. Is that the problem?"),
            ("assistant", "Yes. Each await blocks until that coroutine finishes. Use `asyncio.gather(fetch(a), fetch(b))` to run them concurrently."),
            ("user", "Got it. I also have some pandas processing inside the loop — will that cause issues?"),
            ("assistant", "Yes, pandas is CPU-bound and will block the event loop. Offload it with `loop.run_in_executor`."),
            ("user", "Should I use ThreadPoolExecutor or ProcessPoolExecutor for that?"),
            ("assistant", "ThreadPoolExecutor for I/O-bound tasks; ProcessPoolExecutor for CPU-bound. Pandas falls under CPU-bound."),
            ("user", "I'll switch to ProcessPoolExecutor for the pandas step. Should I create it once globally?"),
            ("assistant", "Yes — create it at startup and reuse it. Creating a new pool per request is expensive."),
            ("user", "Makes sense. I'll refactor the pipeline to use a shared executor."),
            ("assistant", "Also remember to call `executor.shutdown(wait=True)` on app teardown to avoid dangling processes."),
        ],
        [
            ("user", "What's the best approach for testing a module that makes HTTP requests?"),
            ("assistant", "Use `responses` or `pytest-httpx` to mock the HTTP layer, or spin up a real local server with `pytest-httpserver`."),
            ("user", "I've been using `unittest.mock.patch` directly. Is that worse?"),
            ("assistant", "It works but it's fragile — you're patching internals rather than the HTTP boundary. Dedicated libraries are more resilient to library changes."),
            ("user", "My current tests break whenever I upgrade the requests library. That might be why."),
            ("assistant", "Exactly. `responses` intercepts at the transport level, so your tests survive library upgrades."),
            ("user", "Should I also test the actual HTTP calls at some layer, not just the mocked ones?"),
            ("assistant", "Yes — have a small set of integration tests that hit a real test server or a staging environment. Don't rely on mocks alone."),
            ("user", "We don't have a staging environment yet. Would VCR cassettes be a decent intermediate solution?"),
            ("assistant", "VCR (vcrpy) is good for recording real responses and replaying them. Just remember to rotate cassettes when the API changes."),
            ("user", "I'll set up vcrpy for now and push for a proper staging environment later."),
            ("assistant", "Reasonable tradeoff. Make sure cassettes are committed to the repo so CI can replay them."),
        ],
        [
            ("user", "I need to add structured logging to a Python service. What's the recommended setup?"),
            ("assistant", "Use the standard `logging` module with a JSON formatter. `python-json-logger` is a popular lightweight option."),
            ("user", "Should I use the root logger or create a named logger per module?"),
            ("assistant", "Named loggers per module — `logger = logging.getLogger(__name__)`. It gives you fine-grained control over log levels per module."),
            ("user", "How should I configure it? I've seen both `logging.config.dictConfig` and `basicConfig`."),
            ("assistant", "`dictConfig` for anything non-trivial. It keeps the full config in one place and can be loaded from a YAML or dict."),
            ("user", "What about log levels in production vs development?"),
            ("assistant", "Drive it from an environment variable — `LOG_LEVEL=INFO` in prod, `DEBUG` locally. Read it at startup and pass to `dictConfig`."),
            ("user", "Should I log request IDs for tracing across services?"),
            ("assistant", "Absolutely. Use a context variable (Python `contextvars`) to store the request ID and inject it into every log record via a custom filter."),
            ("user", "I'll implement that. One last thing — should log messages be human-readable or always JSON in prod?"),
            ("assistant", "JSON in prod — it's what log aggregators like Datadog or Loki expect. Human-readable in dev for easier debugging."),
        ],
        [
            ("user", "I'm designing a REST API and I'm unsure when to use PUT vs PATCH."),
            ("assistant", "PUT replaces the entire resource; PATCH applies a partial update. Use PATCH when clients only send the fields they want to change."),
            ("user", "So if I have a user resource and only want to update the email, PATCH is better?"),
            ("assistant", "Yes. With PUT the client would need to send the full user object, which is wasteful and error-prone."),
            ("user", "What about idempotency? I've heard both should be idempotent."),
            ("assistant", "PUT is inherently idempotent — applying it twice gives the same result. PATCH *should* be designed to be idempotent, but isn't automatically."),
            ("user", "What does a non-idempotent PATCH look like?"),
            ("assistant", "Something like `PATCH /counter` with body `{increment: 1}` — calling it twice increments twice. Avoid that pattern."),
            ("user", "My API will have PATCH endpoints that set specific fields. That should be idempotent then."),
            ("assistant", "Correct. Setting a field to a value is idempotent; incrementing by a value is not. Stick to the former."),
            ("user", "What status code should I return for a successful PATCH?"),
            ("assistant", "200 with the updated resource is common. 204 No Content is also acceptable if you don't return the body."),
        ],
        [
            ("user", "I need to cache expensive database query results in my Flask app. What's the simplest approach?"),
            ("assistant", "For a single-process app, Flask-Caching with a simple cache backend works well. For multiple workers, use Redis."),
            ("user", "We have multiple gunicorn workers, so Redis it is. How do I set the expiry?"),
            ("assistant", "Set a TTL based on how stale the data can be. For mostly static data, minutes to hours; for dynamic data, seconds."),
            ("user", "How do I handle cache invalidation when the underlying data changes?"),
            ("assistant", "Either invalidate explicitly on write — `cache.delete(key)` — or use short TTLs and accept brief staleness."),
            ("user", "Explicit invalidation sounds safer. But what if the write and the cache delete don't both succeed?"),
            ("assistant", "That's the dual-write problem. A common mitigation is to delete the cache key *after* a successful DB write, inside a try/finally."),
            ("user", "What if the app crashes after the DB write but before the cache delete?"),
            ("assistant", "Then the cache holds stale data until TTL expiry. For most apps that's acceptable. If not, look into transactional outbox patterns."),
            ("user", "For our use case, a short TTL fallback is fine. I'll implement explicit invalidation with a 5-minute TTL as a safety net."),
            ("assistant", "That's a pragmatic approach — belt and suspenders. Works well in practice."),
        ],
        [
            ("user", "I want to use type hints across my project. Is it worth it for a medium-sized codebase?"),
            ("assistant", "Absolutely. Type hints catch whole classes of bugs at development time and massively improve IDE autocomplete."),
            ("user", "Should I annotate everything, or focus on certain parts?"),
            ("assistant", "Start with function signatures — parameters and return types. That gives the most value for the least effort."),
            ("user", "What about complex types like dicts of lists?"),
            ("assistant", "Use `dict[str, list[int]]` in Python 3.9+. For older versions, import from `typing`: `Dict[str, List[int]]`."),
            ("user", "I have a function that can return a value or None. How do I type that?"),
            ("assistant", "Use `Optional[T]` from typing, or the cleaner Python 3.10+ syntax: `T | None`."),
            ("user", "What tool should I use to check the types statically?"),
            ("assistant", "mypy is the standard. For stricter checking with better inference, pyright (used by Pylance in VS Code) is excellent."),
            ("user", "Should I run mypy in CI?"),
            ("assistant", "Yes — add `mypy --strict` (or a configured level) as a CI step. Catches regressions before they merge."),
        ],
        [
            ("user", "I'm refactoring a module that has grown too large. It's over 1000 lines. How should I approach splitting it?"),
            ("assistant", "Look for natural seams — groups of functions that share state or work on the same data. Those are candidates for their own module."),
            ("user", "I have a mix of data models, business logic, and utility functions all in one file."),
            ("assistant", "Classic case. Split into `models.py`, `services.py` (business logic), and `utils.py` (pure utilities)."),
            ("user", "Will I have circular import issues if models and services reference each other?"),
            ("assistant", "Likely yes. The fix is to ensure models don't import from services — only services import from models."),
            ("user", "What if a utility function needs something from models?"),
            ("assistant", "That's a sign it's not really a utility — it belongs in services. Keep utils free of domain dependencies."),
            ("user", "Makes sense. How do I do the refactor safely without breaking everything?"),
            ("assistant", "Move one group at a time, run tests after each move. If you don't have tests, write a few integration tests first as a safety net."),
            ("user", "I have decent test coverage. I'll do it incrementally."),
            ("assistant", "Good approach. Use git commits between each move so you can bisect if something breaks."),
        ],
        [
            ("user", "What's the difference between a context manager and just using try/finally?"),
            ("assistant", "A context manager *is* try/finally, just packaged cleanly. The `with` statement calls `__enter__` and `__exit__` for you."),
            ("user", "When should I write my own context manager instead of using try/finally directly?"),
            ("assistant", "When the setup/teardown pattern repeats in multiple places. Writing a context manager DRYs it up."),
            ("user", "How do I create one? I've seen both the class approach and `contextlib.contextmanager`."),
            ("assistant", "The `@contextmanager` decorator is simpler for one-offs. The class approach is better when you need state or reuse the manager."),
            ("user", "Can you show the decorator version for something like a timer?"),
            ("assistant", "Sure: `@contextmanager def timer(): t0=time.time(); yield; print(time.time()-t0)`. Use it as `with timer():`."),
            ("user", "What happens if an exception is raised inside the `with` block?"),
            ("assistant", "It propagates to the `yield` in your generator. You can catch it there to suppress or log it, then re-raise if needed."),
            ("user", "I'll use the decorator for my db transaction helper. It's only used in a few places."),
            ("assistant", "Perfect use case. Makes the intent clear and keeps your transaction logic out of the calling code."),
        ],
        [
            ("user", "I'm trying to understand Python decorators. I get that they wrap functions, but I'm fuzzy on how."),
            ("assistant", "A decorator is just a function that takes a function and returns a new function. `@mydecorator` is shorthand for `func = mydecorator(func)`."),
            ("user", "So the original function gets replaced?"),
            ("assistant", "Replaced by the wrapper. The wrapper usually calls the original inside, adding behavior before or after."),
            ("user", "How do I preserve the original function's name and docstring?"),
            ("assistant", "Use `@functools.wraps(func)` inside your decorator. Without it, the wrapper's name overwrites the original."),
            ("user", "Can decorators take arguments? Like `@retry(times=3)`?"),
            ("assistant", "Yes — you need a decorator factory: a function that returns a decorator. So it's three levels: factory → decorator → wrapper."),
            ("user", "That sounds complex. Is there a simpler pattern?"),
            ("assistant", "Use `@functools.wraps` and accept `*args, **kwargs` in the wrapper. For parametrized decorators, the nesting is unavoidable but becomes readable once you've seen it a few times."),
            ("user", "I'll write a simple `@log_calls` decorator first to practice."),
            ("assistant", "That's the perfect first decorator. Add `@functools.wraps` from the start so you build the habit."),
        ],
    ],

    # ── EXPERT ──────────────────────────────────────────────────────────────
    "expert": [
        [
            ("user", "I'm profiling a custom CUDA kernel and seeing register spilling at 128 threads per block."),
            ("assistant", "Try `__launch_bounds__(128)` to give the compiler a tighter register budget, or refactor the hot path to reduce live variables."),
            ("user", "Already tried launch_bounds. The bottleneck is the FFT butterfly stage — it has a lot of simultaneous live registers."),
            ("assistant", "Consider radix-4 decomposition over radix-2. It cuts the number of butterfly stages and reduces peak register pressure."),
            ("user", "I prototyped radix-4 and got ~12% improvement, but occupancy is still capped at 50%."),
            ("assistant", "At 50% occupancy the bottleneck is likely latency hiding rather than throughput. Check if you have enough independent instruction streams per warp."),
            ("user", "Nsight shows the primary stall is LG Throttle — load/store queue saturation."),
            ("assistant", "That points to memory bandwidth, not compute. Try a shared-memory transpose between butterfly stages to improve coalescing on the next stage's reads."),
            ("user", "The intermediate data is already in shared mem. The issue is the twiddle factor loads from global memory."),
            ("assistant", "Preload twiddle factors into shared memory or use `__ldg` through the texture cache — twiddle factors are read-only and fit the texture cache well."),
            ("user", "I'll try `__ldg` first since it's a one-line change. If that's not enough I'll do the shared mem preload."),
            ("assistant", "Good prioritization. `__ldg` alone often gives 10-20% on read-only access patterns like this."),
        ],
        [
            ("user", "We're hitting GIL contention in our asyncio service when C extensions acquire the GIL during computation."),
            ("assistant", "The standard fix is to release the GIL in the C extension with `Py_BEGIN_ALLOW_THREADS` around the compute-heavy section."),
            ("user", "We do that for I/O. The problem is NumPy ufunc calls that hold the GIL during the actual computation."),
            ("assistant", "NumPy releases the GIL for most ufuncs by default, but some operations — like fancy indexing — reacquire it. Which operations are you seeing contention on?"),
            ("user", "It's a custom ufunc we wrote that calls into a third-party C library. The library isn't thread-safe so we hold the GIL intentionally."),
            ("assistant", "If the library isn't thread-safe, the options are: serialize calls via a dedicated thread with a queue, or move to a ProcessPoolExecutor to sidestep the GIL entirely."),
            ("user", "ProcessPool adds serialization overhead. We're considering nogil Cython with typed memoryviews as a replacement for the ufunc."),
            ("assistant", "That's the right path. With typed memoryviews you can release the GIL for the entire compute section. Profile the serialization overhead first though — it may be smaller than you expect."),
            ("user", "We also evaluated PyO3 as an alternative to Cython. Thoughts?"),
            ("assistant", "PyO3 is excellent for new code — safer than raw C, great GIL control with `py.allow_threads()`. For existing Cython, the migration cost may not justify it unless you need Rust's safety guarantees."),
            ("user", "We'll benchmark the Cython path first. If it's still insufficient we'll revisit PyO3."),
            ("assistant", "Reasonable. Measure before committing — Cython with nogil and memoryviews can often match hand-written C for this type of workload."),
        ],
        [
            ("user", "I'm designing a distributed rate limiter for our API gateway. Token bucket vs leaky bucket — which fits better?"),
            ("assistant", "Token bucket is generally better for APIs — it allows short bursts up to the bucket capacity while still enforcing an average rate."),
            ("user", "We need to enforce rate limits across multiple gateway replicas. How do I keep the token count consistent?"),
            ("assistant", "Centralize in Redis using atomic Lua scripts or the Redis `INCR` + TTL pattern. The sliding window log is accurate but memory-heavy; sliding window counter is a good tradeoff."),
            ("user", "We're using a fixed window counter now and getting boundary spikes — double the limit at window edges."),
            ("assistant", "Classic fixed window problem. Sliding window counter approximates a true sliding window at a fraction of the memory cost: `count = prev_count * overlap_ratio + curr_count`."),
            ("user", "What's the consistency model here? Can we tolerate slightly stale counts across replicas?"),
            ("assistant", "For rate limiting, slight over-admission is usually acceptable. Use Redis with async replication — the inconsistency window is typically <1ms, far below any rate limit period."),
            ("user", "What about Redis availability? If Redis goes down, should we fail open or closed?"),
            ("assistant", "Depends on your threat model. Fail open (allow requests) preserves availability; fail closed (block requests) prevents abuse. Most production systems fail open with monitoring alerts."),
            ("user", "We'll fail open with a circuit breaker and aggressive alerting. What Redis data structure do you recommend for the sliding window counter?"),
            ("assistant", "A hash per client keyed by time bucket, with a single `EXPIRE` on the key. One `HINCR` per request, one `HGETALL` to compute the window — two round trips, or one with a pipeline."),
        ],
        [
            ("user", "I need to implement a lock-free concurrent hash map for a real-time trading system."),
            ("assistant", "Robin Hood hashing with open addressing and atomic CAS is a proven approach. What's your read/write ratio?"),
            ("user", "Heavily read-skewed — 95% reads. Writes happen on order book updates."),
            ("assistant", "For that ratio, consider a read-copy-update (RCU) style map. Readers are wait-free; writers copy and atomically swap the pointer."),
            ("user", "RCU makes sense, but our map is too large to copy on every write. We need finer granularity."),
            ("assistant", "Segment the map — lock-free segments with RCU within each segment. Write contention is bounded to the segment being updated."),
            ("user", "I'm concerned about ABA on the delete path with our epoch-based memory reclamation."),
            ("assistant", "Epoch-based reclamation is vulnerable to ABA on long-stalled threads. Hazard pointers avoid ABA entirely — a thread publishes the pointer it's accessing, preventing premature reclamation."),
            ("user", "Hazard pointers add a scan cost on every reclamation. At our throughput that might be significant."),
            ("assistant", "Batch retirements and scan periodically rather than per-operation. With R ≈ 2P retired nodes (P = thread count) triggering a scan, amortized cost is O(1) per retirement."),
            ("user", "What's the memory overhead of hazard pointers at 256 threads?"),
            ("assistant", "Typically K hazard pointers per thread (K=2 suffices for most algorithms). So 512 pointer slots — negligible. The retired list per thread is bounded by your scan threshold."),
        ],
        [
            ("user", "We're migrating from a monolith to microservices and struggling with distributed tracing. Where do we start?"),
            ("assistant", "Start with OpenTelemetry — instrument the monolith first with auto-instrumentation, then carry trace context across service boundaries as you split."),
            ("user", "We're using gRPC internally. Does OTel handle context propagation automatically?"),
            ("assistant", "Yes — the OTel gRPC interceptor handles W3C TraceContext propagation. You just need to register the interceptor on client and server."),
            ("user", "What about async boundaries? We use asyncio and the trace context gets lost when we fan out with gather."),
            ("assistant", "Use `context.copy()` before scheduling tasks and restore it inside each coroutine. OTel's asyncio instrumentation handles this with a context propagation hook."),
            ("user", "We're sending traces to Jaeger. At high throughput, what sampling strategy avoids overwhelming the collector?"),
            ("assistant", "Head-based sampling at the root service with a fixed rate (e.g. 1%) for normal traffic. Add tail-based sampling for errors and high-latency requests — Jaeger supports this via the remote sampler."),
            ("user", "Tail-based sampling requires buffering spans. What's the memory tradeoff?"),
            ("assistant", "You buffer spans for the decision window (typically 30s). At high throughput this can be GBs. Use a dedicated collector tier with a span buffer, separate from your application processes."),
            ("user", "We'll start with head-based sampling and add tail-based once we have the collector infrastructure."),
            ("assistant", "Good staged rollout. Head-based is low-overhead and gives you tracing immediately. You can retrofit tail-based decisions later without re-instrumenting."),
        ],
        [
            ("user", "I'm implementing a B-tree storage engine and trying to decide between copy-on-write and in-place updates with WAL."),
            ("assistant", "CoW gives you clean snapshot isolation and crash safety without a WAL, but amplifies write traffic. WAL + in-place is more write-efficient but the recovery path is complex."),
            ("user", "Our workload is write-heavy with frequent range scans. Does that change the calculus?"),
            ("assistant", "For write-heavy workloads, consider an LSM tree instead. B-tree with WAL is fine, but LSM's sequential write pattern tends to outperform B-tree in-place updates at high write throughput."),
            ("user", "We need strong read performance too — LSM read amplification is a concern."),
            ("assistant", "Mitigate LSM read amplification with Bloom filters per SSTable level and aggressive compaction. Leveled compaction (RocksDB-style) gives much better read performance than tiered."),
            ("user", "How does leveled compaction affect write amplification?"),
            ("assistant", "Leveled compaction has higher write amplification (10-30x) vs tiered (3-10x). You're trading write throughput for read performance. For mixed workloads, this is often the right tradeoff."),
            ("user", "We'll implement B-tree with WAL first for simplicity, and profile whether LSM is warranted."),
            ("assistant", "Sensible. WAL recovery is well-understood. If you hit write bottlenecks later, an LSM layer can be added on top of your existing storage interface."),
            ("user", "For the WAL, should I use a ring buffer or append-only segments?"),
            ("assistant", "Append-only segments — simpler crash recovery, easier to truncate after checkpointing, and better for fsync granularity. Ring buffers complicate recovery when the write pointer wraps."),
        ],
        [
            ("user", "I'm evaluating Raft vs Paxos for our consensus layer in a distributed KV store."),
            ("assistant", "Raft is almost always the right choice for new implementations — it's designed for understandability and has better tooling. Paxos is more theoretically general but notoriously hard to implement correctly."),
            ("user", "Multi-Paxos and Raft seem equivalent in practice. What are the real differences?"),
            ("assistant", "Raft mandates a strong leader and has an explicit log matching guarantee. Multi-Paxos leaves leader election and log compaction to the implementer, creating a lot of ambiguity."),
            ("user", "We're concerned about leader bottleneck in Raft under high write throughput."),
            ("assistant", "Address this with batching and pipelining — don't wait for one round-trip before sending the next. Etcd achieves hundreds of thousands of ops/sec this way."),
            ("user", "What about read scalability? All reads going through the leader seems wasteful."),
            ("assistant", "Linearizable reads must go through the leader, but you can serve stale reads from followers. For strong consistency, lease-based reads — where the leader holds a time-bounded lease — avoid a round-trip while staying linearizable."),
            ("user", "How do lease-based reads interact with clock skew?"),
            ("assistant", "The lease duration must be shorter than the minimum clock drift guarantee of your system. Bounded clock skew (e.g. from PTP or TrueTime) lets you use tight leases. With NTP alone, use conservative bounds."),
            ("user", "Our infrastructure uses NTP. What lease duration is safe?"),
            ("assistant", "With NTP, assume up to 100ms drift. Use a lease duration of heartbeat_interval + 2 * max_clock_skew + network_RTT. Err conservatively — a slightly stale read is better than a linearizability violation."),
        ],
        [
            ("user", "We're seeing high tail latency (p99) in our gRPC service even though median latency is fine. Where do I start diagnosing?"),
            ("assistant", "High p99 with low p50 almost always points to one of: GC pauses, thread pool saturation, or lock contention. Start with GC logs and thread pool queue depth metrics."),
            ("user", "We're on Python with grpcio. GC pauses seem unlikely to be the culprit at our scale."),
            ("assistant", "Fair point — CPython GC pauses are usually short. Focus on the thread pool. gRPC Python uses a thread pool for RPC handlers — if it's saturated, new requests queue and p99 spikes."),
            ("user", "How do I check thread pool saturation?"),
            ("assistant", "Instrument `concurrent.futures.ThreadPoolExecutor` with a custom wrapper that tracks queue depth and wait time. Alternatively, profile with `py-spy` during a load test to see thread states."),
            ("user", "py-spy shows most threads are in IO wait during the spike. We're making DB calls per request."),
            ("assistant", "IO wait under load suggests the DB connection pool is exhausted. Add metrics on pool checkout time and pool size. Look for checkout waits in your DB client logs."),
            ("user", "Connection pool max is 20 and we're at 40 concurrent RPC threads. That's clearly the bottleneck."),
            ("assistant", "Either increase the pool size, reduce RPC thread count, or — better — switch to async DB calls so threads aren't blocked during IO."),
            ("user", "Switching to asyncpg and grpc.aio would fix the thread blocking. Is the migration effort worth it?"),
            ("assistant", "For a latency-sensitive service, yes. async gRPC + asyncpg eliminates the thread-per-request model entirely. The migration is a week of work but the tail latency improvement is typically dramatic."),
        ],
        [
            ("user", "I'm implementing a memory allocator for a real-time audio DSP system. What are the key design constraints?"),
            ("assistant", "The primary constraint is deterministic latency — no unbounded allocation time in the audio callback. This rules out general-purpose allocators like glibc malloc."),
            ("user", "We're using a pool allocator with fixed-size blocks. The issue is fragmentation when we need variable-size allocations."),
            ("assistant", "For variable sizes, consider a slab allocator with size classes (e.g. 16, 32, 64, 128 bytes). Objects get the nearest size class — low fragmentation, O(1) alloc/free."),
            ("user", "What about alignment? Our DSP code uses SIMD and needs 32-byte aligned buffers."),
            ("assistant", "Align the slab start to 32 bytes and make all size classes multiples of 32. Alternatively, have a dedicated aligned pool for SIMD buffers."),
            ("user", "The audio callback can't block. How do I handle the case where the pool is exhausted?"),
            ("assistant", "Pre-allocate a reserve pool at startup. If the main pool is exhausted, allocate from the reserve and log a warning — this buys time without dropping audio. Exhausting the reserve is a programming error."),
            ("user", "How do I safely free memory from the audio thread that was allocated on another thread?"),
            ("assistant", "Use a lock-free MPSC queue — the audio thread pushes pointers to free, and a background thread drains the queue and returns blocks to the pool. Audio thread never acquires a lock."),
            ("user", "The MPSC queue itself must be allocation-free. Any recommended implementations?"),
            ("assistant", "A Dmitry Vyukov MPSC queue (linked-list based, one CAS on enqueue) is allocation-free if you pre-allocate nodes. It's wait-free for the producer, which is what matters in the audio callback."),
        ],
        [
            ("user", "We're running a multi-tenant inference service and need to prevent one tenant's request from starving others. What scheduling approaches work here?"),
            ("assistant", "Weighted fair queuing per tenant is the standard approach. Assign each tenant a weight proportional to their quota and drain queues with deficit round-robin."),
            ("user", "We're using batching for GPU efficiency. Batching and fairness seem to conflict."),
            ("assistant", "They do. One approach: batch within a tenant's requests for a time window, then rotate tenants. This amortizes GPU overhead while bounding starvation."),
            ("user", "How do I set the time window? Too long starves others, too short kills batch efficiency."),
            ("assistant", "Make it adaptive — increase the window when the tenant has queued requests, shrink it when the queue is empty. Keep a minimum window floor to ensure batching gains."),
            ("user", "What's a reasonable floor? Our model has a 50ms prefill time."),
            ("assistant", "Floor around 5-10ms — enough to accumulate a batch without being noticeable in tail latency. Profile what batch size maximizes GPU utilization and back-calculate the needed accumulation time."),
            ("user", "We also need to handle priority tiers — premium tenants get lower latency. How does that interact with fairness?"),
            ("assistant", "Use two queues per tenant — high and low priority — and reserve a portion of scheduling slots for high-priority traffic. Weighted fair queuing applies within tiers, strict priority between tiers with anti-starvation for low-priority."),
            ("user", "What's a good anti-starvation mechanism for the low-priority tier?"),
            ("assistant", "Aging: increase a low-priority tenant's effective weight as their wait time grows. When it exceeds a threshold, treat them as high-priority temporarily. Reset the weight after they're scheduled."),
        ],
    ],
}


def _format_conversation(turns: List[Tuple[str, str]]) -> str:
    """Format a list of (role, message) turns into a single string for the model."""
    lines = []
    for role, message in turns:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {message}")
    return "\n".join(lines)


def _format_conversation_chat(turns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Format as a list of chat-style dicts (for tokenizer.apply_chat_template)."""
    return [{"role": role, "content": msg} for role, msg in turns]


def generate_dataset(config: DataConfig) -> Dict:
    """
    Generate a labeled dataset of synthetic expertise conversations.

    Returns:
        {
            "conversations": list of formatted strings,
            "conversations_chat": list of chat-style message lists,
            "labels": {"expertise": list of label strings},
        }
    """
    random.seed(config.seed)

    conversations: List[str] = []
    conversations_chat: List[List[Dict[str, str]]] = []
    labels: List[str] = []

    for level in config.expertise_levels:
        templates = TEMPLATES.get(level, [])
        if not templates:
            print(f"  [WARN] No templates for expertise level '{level}'")
            continue

        for i in range(config.samples_per_class):
            template = templates[i % len(templates)]
            conversations.append(_format_conversation(template))
            conversations_chat.append(_format_conversation_chat(template))
            labels.append(level)

    dataset = {
        "conversations": conversations,
        "conversations_chat": conversations_chat,
        "labels": {"expertise": labels},
    }

    os.makedirs(config.data_dir, exist_ok=True)
    out_path = os.path.join(config.data_dir, "dataset_expertise.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[Dataset] Saved {len(conversations)} conversations to {out_path}")

    return dataset


def prepare_dataset(config: DataConfig) -> Dict:
    """
    Load from cache if available and size matches, otherwise generate fresh.
    """
    cache_path = os.path.join(config.data_dir, "dataset_expertise.json")
    expected = config.samples_per_class * len(config.expertise_levels)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            existing = json.load(f)
        actual = len(existing.get("conversations", []))
        if actual == expected:
            print(f"[Dataset] Loaded cached dataset ({actual} conversations) from {cache_path}")
            return existing
        else:
            print(f"[Dataset] Cache size mismatch (have {actual}, need {expected}). Regenerating ...")

    return generate_dataset(config)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = DataConfig()
    ds = prepare_dataset(cfg)
    print(f"Total conversations: {len(ds['conversations'])}")
    from collections import Counter
    print(f"  expertise: {dict(Counter(ds['labels']['expertise']))}")
    print("\n--- Sample conversation (expert) ---")
    idx = ds["labels"]["expertise"].index("expert")
    print(ds["conversations"][idx][:600])
