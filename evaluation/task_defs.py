#!/usr/bin/env python3
"""Source of truth for the evaluation task set.

Run `python evaluation/task_defs.py` to (re)write tasks.json (what every other script
reads). `reference` solutions live ONLY here: they prove the hidden tests are correct
(see scripts/test_evaluation.py) and are never part of tasks.json or shown to a model.

Each task:
  id            T01..T20
  tier          easy | medium | hard   (rough difficulty for a 3B-7B code model)
  core          True for the 8 tasks of the quick set (run these first)
  request       the text sent to the pipeline / raw model. Written with backticked names,
                numbers and ALL-CAPS terms ON PURPOSE: IFS can only be computed when the
                request contains such "technical entities" (see backend/app/ifs.py).
  entry_points  names the hidden tests need from the generated code
  tests         hidden tests: blocks of Python run with the generated code's names in scope
                (+ helper `raises(Exc, fn, *args)`); a block passes if it raises nothing
"""
import json
from pathlib import Path

TASKS = [
 {"id": "T01", "tier": "easy", "core": True, "entry_points": ["normalize_email"],
  "request": "Write a function `normalize_email(raw)` that lowercases the address, strips surrounding whitespace, removes any `+tag` suffix from the local part (the text between the first `+` and the `@`), and raises `ValueError('invalid email')` if there is no `@`. Include a pytest test.",
  "tests": [
   "assert normalize_email('  Foo+News@Example.COM ') == 'foo@example.com'",
   "assert normalize_email('a@b.com') == 'a@b.com'",
   "assert normalize_email('USER+a+b@Site.org') == 'user@site.org'",
   "assert raises(ValueError, normalize_email, 'no-at-sign')"],
  "reference": '''def normalize_email(raw):
    s = raw.strip().lower()
    if "@" not in s:
        raise ValueError("invalid email")
    local, domain = s.rsplit("@", 1)
    return local.split("+", 1)[0] + "@" + domain
'''},
 {"id": "T02", "tier": "medium", "core": True, "entry_points": ["is_valid_isbn13"],
  "request": "Write `is_valid_isbn13(code)` that accepts hyphens, uses the ISBN-13 checksum (digits weighted 1 and 3 alternately, total divisible by 10), and returns a bool. Include a pytest test.",
  "tests": [
   "assert is_valid_isbn13('978-0-306-40615-7') is True",
   "assert is_valid_isbn13('9780306406157') is True",
   "assert is_valid_isbn13('978-0-306-40615-8') is False",
   "assert is_valid_isbn13('123') is False",
   "assert is_valid_isbn13('978030640615X') is False"],
  "reference": '''def is_valid_isbn13(code):
    s = code.replace("-", "")
    if len(s) != 13 or not s.isdigit():
        return False
    return sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(s)) % 10 == 0
'''},
 {"id": "T03", "tier": "hard", "core": True, "entry_points": ["roman_to_int"],
  "request": "Write `roman_to_int(s)` that converts a Roman numeral between 1 and 3999 (using `I`, `V`, `X`, `L`, `C`, `D`, `M`) to an integer and raises `ValueError` for invalid numerals such as 'IIII'. Include a pytest test.",
  "tests": [
   "assert roman_to_int('III') == 3 and roman_to_int('IV') == 4 and roman_to_int('IX') == 9",
   "assert roman_to_int('LVIII') == 58",
   "assert roman_to_int('MCMXCIV') == 1994",
   "assert roman_to_int('MMMCMXCIX') == 3999",
   "assert raises(ValueError, roman_to_int, 'IIII')",
   "assert raises(ValueError, roman_to_int, 'ABC')",
   "assert raises(ValueError, roman_to_int, '')",
   "assert raises(ValueError, roman_to_int, 'MMMM')"],
  "reference": '''import re

_PAT = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s):
    if not s or not _PAT.match(s):
        raise ValueError("invalid roman numeral")
    total = 0
    for i, ch in enumerate(s):
        v = _VAL[ch]
        total += -v if i + 1 < len(s) and _VAL[s[i + 1]] > v else v
    return total
'''},
 {"id": "T04", "tier": "easy", "core": True, "entry_points": ["chunk_list"],
  "request": "Write `chunk_list(items, size)` that splits a list into consecutive chunks of `size` and returns a list of lists (the last chunk may be shorter). Raise `ValueError` if `size` < 1. Include a pytest test.",
  "tests": [
   "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
   "assert chunk_list([], 3) == []",
   "assert chunk_list([1, 2], 5) == [[1, 2]]",
   "assert chunk_list(list('abcdef'), 3) == [['a', 'b', 'c'], ['d', 'e', 'f']]",
   "assert raises(ValueError, chunk_list, [1, 2, 3], 0)"],
  "reference": '''def chunk_list(items, size):
    if size < 1:
        raise ValueError("size must be >= 1")
    return [list(items[i:i + size]) for i in range(0, len(items), size)]
'''},
 {"id": "T05", "tier": "hard", "core": True, "entry_points": ["merge_intervals"],
  "request": "Write `merge_intervals(intervals)` that takes a list of `[start, end]` pairs, merges every pair that overlaps or touches (end == next start), and returns a list of `[start, end]` lists sorted by start. Include a pytest test.",
  "tests": [
   "assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]",
   "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]",
   "assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]",
   "assert merge_intervals([]) == []",
   "assert merge_intervals([[5, 7]]) == [[5, 7]]",
   "assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]"],
  "reference": '''def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
'''},
 {"id": "T06", "tier": "hard", "core": False, "entry_points": ["LRUCache"],
  "request": "Write `class LRUCache` with `__init__(self, capacity)`, `get(key)` that returns the value or -1 when the key is missing, and `put(key, value)` that evicts the least recently used key when `capacity` is exceeded. Both `get` and `put` count as a use. Include a pytest test.",
  "tests": [
   "c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); assert c.get(2) == -1; c.put(4, 4); assert c.get(1) == -1; assert c.get(3) == 3; assert c.get(4) == 4",
   "c = LRUCache(1); c.put(1, 1); c.put(1, 5); assert c.get(1) == 5; c.put(2, 2); assert c.get(1) == -1",
   "c = LRUCache(2); assert c.get(9) == -1",
   "c = LRUCache(2); c.put(1, 'a'); c.put(2, 'b'); c.put(1, 'c'); c.put(3, 'd'); assert c.get(2) == -1 and c.get(1) == 'c'"],
  "reference": '''from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return -1
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)
'''},
 {"id": "T07", "tier": "easy", "core": True, "entry_points": ["validate_password"],
  "request": "Write `validate_password(pw)` that returns True only if the password has at least 12 characters, an uppercase letter, a lowercase letter, a digit and at least one of the symbols `!@#$`; otherwise return False. Include a pytest test.",
  "tests": [
   "assert validate_password('Abcdefghij1!') is True",
   "assert validate_password('Abcdefghij1') is False",
   "assert validate_password('abcdefghij1!') is False",
   "assert validate_password('ABCDEFGHIJ1!') is False",
   "assert validate_password('Abcdefghijk!') is False",
   "assert validate_password('Abcdefghij12') is False",
   "assert validate_password('Abcdefghij1%') is False",
   "assert validate_password('Short1!a') is False"],
  "reference": '''def validate_password(pw):
    return (len(pw) >= 12 and any(c.isupper() for c in pw) and any(c.islower() for c in pw)
            and any(c.isdigit() for c in pw) and any(c in "!@#$" for c in pw))
'''},
 {"id": "T08", "tier": "medium", "core": True, "entry_points": ["parse_iso_date"],
  "request": "Write `parse_iso_date(s)` that parses a `YYYY-MM-DD` string into a `datetime.date` and raises `ValueError` for a month outside 1-12, a day that does not exist in that month, or any other malformed string. Include a pytest test.",
  "tests": [
   "import datetime; assert parse_iso_date('2024-02-29') == datetime.date(2024, 2, 29)",
   "import datetime; assert parse_iso_date('1999-12-31') == datetime.date(1999, 12, 31)",
   "assert raises(ValueError, parse_iso_date, '2023-02-29')",
   "assert raises(ValueError, parse_iso_date, '2024-13-01')",
   "assert raises(ValueError, parse_iso_date, '2024-00-10')",
   "assert raises(ValueError, parse_iso_date, '20240101')",
   "assert raises(ValueError, parse_iso_date, 'hello')"],
  "reference": '''import datetime
import re


def parse_iso_date(s):
    m = re.fullmatch(r"(\\d{4})-(\\d{2})-(\\d{2})", s)
    if not m:
        raise ValueError("malformed date")
    return datetime.date(int(m[1]), int(m[2]), int(m[3]))
'''},
 {"id": "T09", "tier": "medium", "core": False, "entry_points": ["celsius_to_fahrenheit", "fahrenheit_to_celsius"],
  "request": "Write `celsius_to_fahrenheit(c)` and `fahrenheit_to_celsius(f)` that round the result to 2 decimals and raise `ValueError` for temperatures below absolute zero (-273.15 C / -459.67 F). Include a pytest test.",
  "tests": [
   "assert celsius_to_fahrenheit(100) == 212.0 and celsius_to_fahrenheit(0) == 32.0",
   "assert celsius_to_fahrenheit(37) == 98.6",
   "assert fahrenheit_to_celsius(98.6) == 37.0 and fahrenheit_to_celsius(32) == 0.0",
   "assert celsius_to_fahrenheit(-273.15) == -459.67",
   "assert raises(ValueError, celsius_to_fahrenheit, -273.16)",
   "assert raises(ValueError, fahrenheit_to_celsius, -459.68)"],
  "reference": '''def celsius_to_fahrenheit(c):
    if c < -273.15:
        raise ValueError("below absolute zero")
    return round(c * 9 / 5 + 32, 2)


def fahrenheit_to_celsius(f):
    if f < -459.67:
        raise ValueError("below absolute zero")
    return round((f - 32) * 5 / 9, 2)
'''},
 {"id": "T10", "tier": "medium", "core": False, "entry_points": ["word_frequency"],
  "request": "Write `word_frequency(text, top_n=3)` that lowercases the text, ignores punctuation, and returns a list of `(word, count)` tuples for the `top_n` most frequent words, ordered by count descending and then alphabetically. Include a pytest test.",
  "tests": [
   "assert word_frequency('the cat and the hat and the bat') == [('the', 3), ('and', 2), ('bat', 1)]",
   "assert word_frequency('Hello, hello! World.', top_n=2) == [('hello', 2), ('world', 1)]",
   "assert word_frequency('') == []",
   "assert word_frequency('b a', top_n=10) == [('a', 1), ('b', 1)]"],
  "reference": '''import re
from collections import Counter


def word_frequency(text, top_n=3):
    counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
'''},
 {"id": "T11", "tier": "medium", "core": False, "entry_points": ["camel_to_snake"],
  "request": "Write `camel_to_snake(name)` that converts `camelCase` and `PascalCase` identifiers to `snake_case`, keeping acronyms together (`HTTPResponseCode` -> `http_response_code`). Include a pytest test.",
  "tests": [
   "assert camel_to_snake('camelCase') == 'camel_case'",
   "assert camel_to_snake('HTTPResponseCode') == 'http_response_code'",
   "assert camel_to_snake('already_snake') == 'already_snake'",
   "assert camel_to_snake('XMLParser') == 'xml_parser'",
   "assert camel_to_snake('simple') == 'simple'",
   "assert camel_to_snake('getHTTPResponse') == 'get_http_response'"],
  "reference": '''import re


def camel_to_snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()
'''},
 {"id": "T12", "tier": "medium", "core": False, "entry_points": ["is_balanced"],
  "request": "Write `is_balanced(s)` that returns True if the brackets `()`, `[]` and `{}` in the string are balanced and correctly nested, ignoring every other character. Include a pytest test.",
  "tests": [
   "assert is_balanced('([]{})') is True",
   "assert is_balanced('([)]') is False",
   "assert is_balanced('((') is False",
   "assert is_balanced('') is True",
   "assert is_balanced('a(b)c[d]{e}') is True",
   "assert is_balanced(')(') is False"],
  "reference": '''def is_balanced(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
'''},
 {"id": "T13", "tier": "easy", "core": True, "entry_points": ["fizzbuzz"],
  "request": "Write `fizzbuzz(n)` that returns a list of strings for the numbers 1 to `n`: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, otherwise the number as a string. Return `[]` for `n` = 0 and raise `ValueError` for negative `n`. Include a pytest test.",
  "tests": [
   "assert fizzbuzz(15) == ['1','2','Fizz','4','Buzz','Fizz','7','8','Fizz','Buzz','11','Fizz','13','14','FizzBuzz']",
   "assert fizzbuzz(0) == []",
   "assert fizzbuzz(5)[-1] == 'Buzz' and fizzbuzz(1) == ['1']",
   "assert raises(ValueError, fizzbuzz, -1)"],
  "reference": '''def fizzbuzz(n):
    if n < 0:
        raise ValueError("n must be >= 0")
    out = []
    for i in range(1, n + 1):
        out.append("FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i))
    return out
'''},
 {"id": "T14", "tier": "easy", "core": False, "entry_points": ["binary_search"],
  "request": "Write `binary_search(items, target)` that returns the index of `target` in the sorted list `items`, or -1 if it is not present. Include a pytest test.",
  "tests": [
   "assert binary_search([1, 3, 5, 7, 9], 7) == 3",
   "assert binary_search([1, 3, 5, 7, 9], 1) == 0 and binary_search([1, 3, 5, 7, 9], 9) == 4",
   "assert binary_search([1, 3, 5, 7, 9], 4) == -1",
   "assert binary_search([], 1) == -1",
   "assert binary_search([5], 5) == 0"],
  "reference": '''def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''},
 {"id": "T15", "tier": "medium", "core": False, "entry_points": ["flatten"],
  "request": "Write `flatten(nested)` that recursively flattens arbitrarily nested lists into a single flat list, keeping strings whole (do not split 'ab' into characters). Include a pytest test.",
  "tests": [
   "assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]",
   "assert flatten(['ab', ['cd']]) == ['ab', 'cd']",
   "assert flatten([]) == [] and flatten([[], [[]]]) == []",
   "assert flatten([1, 2, 3]) == [1, 2, 3]"],
  "reference": '''def flatten(nested):
    out = []
    for x in nested:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
'''},
 {"id": "T16", "tier": "easy", "core": False, "entry_points": ["dedupe_preserve_order"],
  "request": "Write `dedupe_preserve_order(items)` that returns a new list with duplicates removed, keeping the first occurrence of each item in its original order. Include a pytest test.",
  "tests": [
   "assert dedupe_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]",
   "assert dedupe_preserve_order([]) == []",
   "assert dedupe_preserve_order(list('banana')) == ['b', 'a', 'n']",
   "x = [1, 1, 2]; y = dedupe_preserve_order(x); assert x == [1, 1, 2] and y == [1, 2]"],
  "reference": '''def dedupe_preserve_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
'''},
 {"id": "T17", "tier": "hard", "core": False, "entry_points": ["MinStack"],
  "request": "Write `class MinStack` with `push(x)`, `pop()` (removes and returns the top value), `top()` and `get_min()`, all O(1); `pop()` and `top()` on an empty stack raise `IndexError`. `get_min()` returns the smallest value currently on the stack. Include a pytest test.",
  "tests": [
   "s = MinStack(); s.push(5); s.push(3); s.push(7); assert s.get_min() == 3 and s.top() == 7",
   "s = MinStack(); s.push(5); s.push(3); s.push(7); assert s.pop() == 7 and s.pop() == 3 and s.get_min() == 5 and s.top() == 5",
   "s = MinStack(); assert raises(IndexError, s.pop)",
   "s = MinStack(); assert raises(IndexError, s.top)",
   "s = MinStack(); s.push(2); s.push(2); s.pop(); assert s.get_min() == 2"],
  "reference": '''class MinStack:
    def __init__(self):
        self._s = []
        self._m = []

    def push(self, x):
        self._s.append(x)
        self._m.append(x if not self._m else min(x, self._m[-1]))

    def pop(self):
        if not self._s:
            raise IndexError("pop from empty stack")
        self._m.pop()
        return self._s.pop()

    def top(self):
        if not self._s:
            raise IndexError("top of empty stack")
        return self._s[-1]

    def get_min(self):
        return self._m[-1]
'''},
 {"id": "T18", "tier": "medium", "core": False, "entry_points": ["luhn_check"],
  "request": "Write `luhn_check(number)` that returns True if a string of digits (spaces allowed) passes the Luhn checksum and False otherwise, including for any string containing other characters. Include a pytest test.",
  "tests": [
   "assert luhn_check('79927398713') is True",
   "assert luhn_check('79927398710') is False",
   "assert luhn_check('4539 1488 0343 6467') is True",
   "assert luhn_check('4539 1488 0343 6468') is False",
   "assert luhn_check('abc') is False"],
  "reference": '''def luhn_check(number):
    s = number.replace(" ", "")
    if not s or not s.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(s)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0
'''},
 {"id": "T19", "tier": "easy", "core": False, "entry_points": ["slugify"],
  "request": "Write `slugify(text)` that lowercases the text, replaces every run of non-alphanumeric characters with a single `-`, and strips leading and trailing `-`. Include a pytest test.",
  "tests": [
   "assert slugify('Hello, World!') == 'hello-world'",
   "assert slugify('  Multiple   spaces  ') == 'multiple-spaces'",
   "assert slugify('C++ & Python 3.12') == 'c-python-3-12'",
   "assert slugify('---') == ''",
   "assert slugify('Already-slugged') == 'already-slugged'"],
  "reference": '''import re


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
'''},
 {"id": "T20", "tier": "hard", "core": False, "entry_points": ["parse_csv_line"],
  "request": "Write `parse_csv_line(line)` that splits one CSV line into a list of fields, honouring double-quoted fields that contain commas and treating a doubled quote `\"\"` inside quotes as a literal quote. Include a pytest test.",
  "tests": [
   "assert parse_csv_line('a,b,c') == ['a', 'b', 'c']",
   "assert parse_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']",
   "assert parse_csv_line('\"say \"\"hi\"\"\",x') == ['say \"hi\"', 'x']",
   "assert parse_csv_line('') == ['']",
   "assert parse_csv_line('a,,c') == ['a', '', 'c']"],
  "reference": '''def parse_csv_line(line):
    fields, cur, in_q, i = [], [], False, 0
    while i < len(line):
        ch = line[i]
        if in_q:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    in_q = False
            else:
                cur.append(ch)
        elif ch == '"':
            in_q = True
        elif ch == ",":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    fields.append("".join(cur))
    return fields
'''},
]


def public_tasks() -> list[dict]:
    return [{k: v for k, v in t.items() if k != "reference"} for t in TASKS]


if __name__ == "__main__":
    out = Path(__file__).with_name("tasks.json")
    out.write_text(json.dumps({"version": 1, "tasks": public_tasks()}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(TASKS)} tasks, {sum(t['core'] for t in TASKS)} core)")
