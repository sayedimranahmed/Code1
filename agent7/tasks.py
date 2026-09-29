"""Benchmark: 50 Python coding tasks with visible examples and hidden tests.

The agent sees `prompt` and `examples`. It never sees `tests` or `reference`.
`reference` exists only so `python -m agent7 validate` can prove every hidden
test suite is satisfiable and rejects a do-nothing stub.

Split: tasks at index i with i % 5 in (1, 3) are held-out (20), the rest are
train (30). The optimizer only ever learns from train; held-out is read once
per run to confirm that a gain generalizes.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    entry: str
    prompt: str
    examples: tuple = ()
    tests: tuple = ()
    reference: str = ""
    split: str = field(default="train")


_RAW = []


def T(id, entry, prompt, examples, tests, reference):
    _RAW.append((id, entry, prompt.strip(), tuple(examples), tuple(tests), reference.strip() + "\n"))


T("slugify", "slugify",
  """Write `slugify(s: str) -> str`. Lowercase the string, replace every run of
characters that are not ASCII letters or digits with a single '-', and strip
leading/trailing '-'. An input with no letters or digits returns ''.""",
  ["assert slugify('Hello World') == 'hello-world'"],
  ["assert slugify('  Hello,   World!! ') == 'hello-world'",
   "assert slugify('---') == ''",
   "assert slugify('') == ''",
   "assert slugify('Café au lait') == 'caf-au-lait'",
   "assert slugify('a__b--c') == 'a-b-c'",
   "assert slugify('ABC123') == 'abc123'"],
  """
import re
def slugify(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')
""")

T("roman_to_int", "roman_to_int",
  """Write `roman_to_int(s: str) -> int` converting a valid Roman numeral
(1..3999, subtractive notation) to an integer.""",
  ["assert roman_to_int('XIV') == 14"],
  ["assert roman_to_int('I') == 1",
   "assert roman_to_int('MCMXCIV') == 1994",
   "assert roman_to_int('MMMCMXCIX') == 3999",
   "assert roman_to_int('XL') == 40",
   "assert roman_to_int('CDXLIV') == 444"],
  """
def roman_to_int(s):
    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and v[c] < v[s[i+1]]:
            total -= v[c]
        else:
            total += v[c]
    return total
""")

T("int_to_roman", "int_to_roman",
  """Write `int_to_roman(n: int) -> str` for 1 <= n <= 3999 using standard
subtractive notation. Raise ValueError for any n outside that range.""",
  ["assert int_to_roman(9) == 'IX'"],
  ["assert int_to_roman(1994) == 'MCMXCIV'",
   "assert int_to_roman(3999) == 'MMMCMXCIX'",
   "assert int_to_roman(4) == 'IV'",
   "assert int_to_roman(400) == 'CD'",
   "V = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\ndef back(r):\n    return sum(-V[c] if i + 1 < len(r) and V[c] < V[r[i+1]] else V[c] for i, c in enumerate(r))\nfor n in range(1, 4000):\n    r = int_to_roman(n)\n    assert back(r) == n and len(r) == len(int_to_roman(n)), n\nassert int_to_roman(49) == 'XLIX' and int_to_roman(500) == 'D' and int_to_roman(50) == 'L'",
   "try:\n    int_to_roman(0)\n    raise AssertionError('no error')\nexcept ValueError:\n    pass",
   "try:\n    int_to_roman(4000)\n    raise AssertionError('no error')\nexcept ValueError:\n    pass"],
  """
def int_to_roman(n):
    if not 1 <= n <= 3999:
        raise ValueError(n)
    pairs = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
             (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = []
    for v, s in pairs:
        while n >= v:
            out.append(s)
            n -= v
    return ''.join(out)
""")

T("is_balanced", "is_balanced",
  """Write `is_balanced(s: str) -> bool`. Return True if the brackets (), [] and
{} in s are balanced and properly nested. All other characters are ignored.""",
  ["assert is_balanced('a(b[c]{d})') is True"],
  ["assert is_balanced('') is True",
   "assert is_balanced('(]') is False",
   "assert is_balanced('([)]') is False",
   "assert is_balanced('((') is False",
   "assert is_balanced('))((') is False",
   "assert is_balanced('x{y[z(w)]}') is True"],
  """
def is_balanced(s):
    pairs = {')':'(', ']':'[', '}':'{'}
    st = []
    for c in s:
        if c in '([{':
            st.append(c)
        elif c in pairs:
            if not st or st.pop() != pairs[c]:
                return False
    return not st
""")

T("rle_encode", "rle_encode",
  """Write `rle_encode(s: str) -> str`. Run-length encode s as count followed by
character for every run, e.g. 'aaabcc' -> '3a1b2c'. Empty input gives ''.""",
  ["assert rle_encode('aaabcc') == '3a1b2c'"],
  ["assert rle_encode('') == ''",
   "assert rle_encode('a') == '1a'",
   "assert rle_encode('a' * 12 + 'b') == '12a1b'",
   "assert rle_encode('abab') == '1a1b1a1b'",
   "assert rle_encode('  ') == '2 '"],
  """
def rle_encode(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f'{j - i}{s[i]}')
        i = j
    return ''.join(out)
""")

T("rle_decode", "rle_decode",
  """Write `rle_decode(s: str) -> str`, the inverse of count-then-char run-length
encoding: '3a1b2c' -> 'aaabcc'. Counts can have several digits. Encoded
characters are never digits.""",
  ["assert rle_decode('3a1b2c') == 'aaabcc'"],
  ["assert rle_decode('') == ''",
   "assert rle_decode('12a1b') == 'a' * 12 + 'b'",
   "assert rle_decode('1 2x') == ' xx'",
   "assert rle_decode('100z') == 'z' * 100"],
  """
def rle_decode(s):
    out, num = [], ''
    for c in s:
        if c.isdigit():
            num += c
        else:
            out.append(c * int(num))
            num = ''
    return ''.join(out)
""")

T("merge_intervals", "merge_intervals",
  """Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]`.
Merge overlapping intervals [start, end] (inclusive). Intervals that touch
(end == next start) are merged. Input may be unsorted; output is sorted by
start. Do not mutate the input.""",
  ["assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]"],
  ["assert merge_intervals([]) == []",
   "assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]",
   "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
   "assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]",
   "x = [[3,4],[1,2]]\nmerge_intervals(x)\nassert x == [[3,4],[1,2]]"],
  """
def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
""")

T("top_k_frequent", "top_k_frequent",
  """Write `top_k_frequent(words: list[str], k: int) -> list[str]` returning the k
most frequent words, ordered by frequency descending and then alphabetically.
If there are fewer than k distinct words, return all of them.""",
  ["assert top_k_frequent(['a','b','a'], 1) == ['a']"],
  ["assert top_k_frequent(['i','love','leetcode','i','love','coding'], 2) == ['i','love']",
   "assert top_k_frequent(['b','a','c'], 2) == ['a','b']",
   "assert top_k_frequent([], 3) == []",
   "assert top_k_frequent(['x','y'], 5) == ['x','y']"],
  """
from collections import Counter
def top_k_frequent(words, k):
    c = Counter(words)
    return sorted(c, key=lambda w: (-c[w], w))[:k]
""")

T("flatten", "flatten",
  """Write `flatten(x) -> list` that flattens arbitrarily nested lists and tuples
into a flat list, preserving order. Strings, bytes and dicts are atoms and
must not be iterated into.""",
  ["assert flatten([1, [2, (3, 4)], 5]) == [1, 2, 3, 4, 5]"],
  ["assert flatten([]) == []",
   "assert flatten([['ab'], 'cd']) == ['ab', 'cd']",
   "assert flatten([[[[]]]]) == []",
   "assert flatten([{'a': 1}, [b'x']]) == [{'a': 1}, b'x']",
   "assert flatten([[[[[[[[[[[[[[1]]]]]]]]]]]]]]) == [1]"],
  """
def flatten(x):
    out = []
    def go(v):
        if isinstance(v, (list, tuple)):
            for i in v:
                go(i)
        else:
            out.append(v)
    go(x)
    return out
""")

T("lru_cache", "LRUCache",
  """Write a class `LRUCache(capacity: int)` with `get(key) -> value` (returns -1
if missing) and `put(key, value)`. When inserting beyond capacity, evict the
least recently used key. Both get and put count as a use.""",
  ["c = LRUCache(1)\nc.put(1, 1)\nassert c.get(1) == 1"],
  ["c = LRUCache(2)\nc.put(1,1); c.put(2,2)\nassert c.get(1) == 1\nc.put(3,3)\nassert c.get(2) == -1\nassert c.get(3) == 3",
   "c = LRUCache(2)\nc.put(1,1); c.put(2,2); c.put(1,10); c.put(3,3)\nassert c.get(1) == 10\nassert c.get(2) == -1",
   "c = LRUCache(1)\nassert c.get(5) == -1"],
  """
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)
""")

T("parse_duration", "parse_duration",
  """Write `parse_duration(s: str) -> int` returning total seconds for strings
like '1h30m', '2d', '45s', '1d2h3m4s'. Units are d, h, m, s, each at most once,
in that order, each preceded by a non-negative integer. Raise ValueError for
empty input or anything else malformed (unknown unit, wrong order, repeated
unit, missing number, stray characters).""",
  ["assert parse_duration('1h30m') == 5400"],
  ["assert parse_duration('2d') == 172800",
   "assert parse_duration('1d2h3m4s') == 93784",
   "assert parse_duration('0s') == 0",
   "for bad in ['', 'h', '1x', '1m1h', '1h1h', '1h 2m', '-1s', '5']:\n    try:\n        parse_duration(bad)\n        raise AssertionError(bad)\n    except ValueError:\n        pass"],
  """
import re
def parse_duration(s):
    m = re.fullmatch(r'(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', s)
    if not s or not m:
        raise ValueError(s)
    d, h, mi, se = (int(g) if g else 0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
""")

T("camel_to_snake", "camel_to_snake",
  """Write `camel_to_snake(name: str) -> str` converting CamelCase or camelCase to
snake_case. Acronyms stay together: 'HTTPServerError' -> 'http_server_error',
'getHTTPResponseCode' -> 'get_http_response_code'. Digits attach to the
preceding word: 'version2Update' -> 'version2_update'.""",
  ["assert camel_to_snake('CamelCase') == 'camel_case'"],
  ["assert camel_to_snake('HTTPServerError') == 'http_server_error'",
   "assert camel_to_snake('getHTTPResponseCode') == 'get_http_response_code'",
   "assert camel_to_snake('version2Update') == 'version2_update'",
   "assert camel_to_snake('already_snake') == 'already_snake'",
   "assert camel_to_snake('A') == 'a'",
   "assert camel_to_snake('XMLParser') == 'xml_parser'"],
  """
import re
def camel_to_snake(name):
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\\1_\\2', name)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s)
    return s.lower()
""")

T("word_wrap", "word_wrap",
  """Write `word_wrap(text: str, width: int) -> list[str]`. Split text on
whitespace and greedily pack words into lines of at most `width` characters,
joined by single spaces. A word longer than width is split into chunks of
exactly width characters (the last chunk may be shorter); chunks are placed
on their own lines and the following word starts a new line. Empty or
whitespace-only text gives [].""",
  ["assert word_wrap('the quick brown fox', 10) == ['the quick', 'brown fox']"],
  ["assert word_wrap('', 5) == []",
   "assert word_wrap('   ', 5) == []",
   "assert word_wrap('abcdefghij xy', 4) == ['abcd', 'efgh', 'ij', 'xy']",
   "assert word_wrap('a b c d', 3) == ['a b', 'c d']",
   "assert word_wrap('hello', 5) == ['hello']"],
  """
def word_wrap(text, width):
    lines, cur = [], ''
    for w in text.split():
        if len(w) > width:
            if cur:
                lines.append(cur)
                cur = ''
            for i in range(0, len(w), width):
                lines.append(w[i:i+width])
            continue
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += ' ' + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
""")

T("bisect_left_index", "find_first",
  """Write `find_first(a: list[int], x: int) -> int` returning the index of the
first occurrence of x in the sorted list a, or -1 if absent. Must be
O(log n); it will be tested on a list of a million elements.""",
  ["assert find_first([1,2,2,2,3], 2) == 1"],
  ["assert find_first([], 1) == -1",
   "assert find_first([1,3,5], 4) == -1",
   "assert find_first([7,7,7], 7) == 0",
   "a = [0]*500000 + [1]*500000\nimport time\nt=time.time()\nfor _ in range(2000): r = find_first(a, 1)\nassert r == 500000 and time.time()-t < 1.0",
   "assert find_first([1,2,3], 3) == 2"],
  """
import bisect
def find_first(a, x):
    i = bisect.bisect_left(a, x)
    return i if i < len(a) and a[i] == x else -1
""")

T("spiral_order", "spiral_order",
  """Write `spiral_order(m: list[list[int]]) -> list[int]` returning the elements
of a rectangular matrix in clockwise spiral order starting top-left.""",
  ["assert spiral_order([[1,2],[3,4]]) == [1,2,4,3]"],
  ["assert spiral_order([]) == []",
   "assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
   "assert spiral_order([[1,2,3,4]]) == [1,2,3,4]",
   "assert spiral_order([[1],[2],[3]]) == [1,2,3]",
   "assert spiral_order([[1,2,3],[4,5,6]]) == [1,2,3,6,5,4]"],
  """
def spiral_order(m):
    out = []
    m = [list(r) for r in m]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out
""")

T("rotate_matrix", "rotate",
  """Write `rotate(m: list[list[int]]) -> list[list[int]]` returning a new matrix
rotated 90 degrees clockwise. The matrix may be rectangular. Do not mutate
the input.""",
  ["assert rotate([[1,2],[3,4]]) == [[3,1],[4,2]]"],
  ["assert rotate([[1,2,3]]) == [[1],[2],[3]]",
   "assert rotate([[1],[2],[3]]) == [[3,2,1]]",
   "assert rotate([]) == []",
   "x=[[1,2],[3,4]]\nrotate(x)\nassert x == [[1,2],[3,4]]"],
  """
def rotate(m):
    return [list(r) for r in zip(*m[::-1])]
""")

T("dijkstra", "shortest_paths",
  """Write `shortest_paths(graph: dict, src) -> dict`. graph maps node -> list of
(neighbor, weight) with non-negative weights. Return a dict of shortest
distance from src to every reachable node (src maps to 0). Unreachable nodes
are omitted. Nodes may appear only as neighbors, never as keys.""",
  ["assert shortest_paths({'a': [('b', 1)]}, 'a') == {'a': 0, 'b': 1}"],
  ["g={'a':[('b',4),('c',1)],'c':[('b',1)],'b':[('d',1)]}\nassert shortest_paths(g,'a') == {'a':0,'c':1,'b':2,'d':3}",
   "assert shortest_paths({}, 'x') == {'x': 0}",
   "g={'a':[('b',1)],'z':[('a',1)]}\nassert shortest_paths(g,'a') == {'a':0,'b':1}",
   "g={'a':[('a',0),('b',0)]}\nassert shortest_paths(g,'a') == {'a':0,'b':0}"],
  """
import heapq
def shortest_paths(graph, src):
    dist = {src: 0}
    pq = [(0, 0, src)]
    n = 0
    done = set()
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, w in graph.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                n += 1
                heapq.heappush(pq, (nd, n, v))
    return dist
""")

T("topo_sort", "topo_sort",
  """Write `topo_sort(deps: dict[str, list[str]]) -> list[str]`. deps maps a node
to the nodes it depends on (which must come before it). Nodes that appear
only as dependencies are included too. Among nodes that are ready at the same
time, pick the alphabetically smallest first. Raise ValueError on a cycle.""",
  ["assert topo_sort({'b': ['a']}) == ['a', 'b']"],
  ["assert topo_sort({}) == []",
   "assert topo_sort({'c': [], 'b': [], 'a': []}) == ['a','b','c']",
   "assert topo_sort({'d':['b','c'],'b':['a'],'c':['a']}) == ['a','b','c','d']",
   "assert topo_sort({'a':['z'],'y':[]}) == ['y','z','a']",
   "try:\n    topo_sort({'a':['b'],'b':['a']})\n    raise AssertionError()\nexcept ValueError:\n    pass"],
  """
import heapq
def topo_sort(deps):
    nodes = set(deps)
    for v in deps.values():
        nodes.update(v)
    indeg = {n: 0 for n in nodes}
    out_edges = {n: [] for n in nodes}
    for n, ds in deps.items():
        for d in set(ds):
            indeg[n] += 1
            out_edges[d].append(n)
    ready = [n for n in nodes if indeg[n] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for m in out_edges[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(ready, m)
    if len(order) != len(nodes):
        raise ValueError('cycle')
    return order
""")

T("lcs_length", "lcs_length",
  """Write `lcs_length(a: str, b: str) -> int` returning the length of the
longest common subsequence. Inputs can be up to 2000 characters each and must
finish in a few seconds.""",
  ["assert lcs_length('abcde', 'ace') == 3"],
  ["assert lcs_length('', 'abc') == 0",
   "assert lcs_length('abc', 'def') == 0",
   "assert lcs_length('AGGTAB', 'GXTXAYB') == 4",
   "assert lcs_length('a'*1500, 'a'*1500) == 1500"],
  """
def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j+1], cur[j]))
        prev = cur
    return prev[-1]
""")

T("edit_distance", "edit_distance",
  """Write `edit_distance(a: str, b: str) -> int`, the Levenshtein distance
(insert, delete, substitute each cost 1).""",
  ["assert edit_distance('kitten', 'sitting') == 3"],
  ["assert edit_distance('', '') == 0",
   "assert edit_distance('', 'abc') == 3",
   "assert edit_distance('flaw', 'lawn') == 2",
   "assert edit_distance('same', 'same') == 0",
   "assert edit_distance('intention', 'execution') == 5"],
  """
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]
""")

T("valid_ipv4", "is_valid_ipv4",
  """Write `is_valid_ipv4(s: str) -> bool`. Valid addresses are four decimal
octets 0-255 separated by dots, with no leading zeros (except '0' itself), no
signs, no whitespace, and ASCII digits only.""",
  ["assert is_valid_ipv4('192.168.0.1') is True"],
  ["assert is_valid_ipv4('255.255.255.255') is True",
   "assert is_valid_ipv4('0.0.0.0') is True",
   "assert is_valid_ipv4('256.1.1.1') is False",
   "assert is_valid_ipv4('01.1.1.1') is False",
   "assert is_valid_ipv4('1.1.1') is False",
   "assert is_valid_ipv4('1.1.1.1.') is False",
   "assert is_valid_ipv4(' 1.1.1.1') is False",
   "assert is_valid_ipv4('1.+1.1.1') is False",
   "assert is_valid_ipv4('1.1.1.\\u0661') is False"],
  """
def is_valid_ipv4(s):
    parts = s.split('.')
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or not all(c in '0123456789' for c in p):
            return False
        if len(p) > 1 and p[0] == '0':
            return False
        if int(p) > 255:
            return False
    return True
""")

T("parse_csv_line", "parse_csv_line",
  """Write `parse_csv_line(line: str) -> list[str]` parsing a single CSV line.
Fields are separated by commas. A field may be wrapped in double quotes, in
which case it may contain commas, and a doubled quote "" inside it means a
literal quote. Do not use the csv module. An empty line yields [''].""",
  ["assert parse_csv_line('a,b,c') == ['a', 'b', 'c']"],
  ["assert parse_csv_line('') == ['']",
   "assert parse_csv_line('a,,c') == ['a', '', 'c']",
   "assert parse_csv_line('\"x,y\",z') == ['x,y', 'z']",
   "assert parse_csv_line('\"he said \"\"hi\"\"\",2') == ['he said \"hi\"', '2']",
   "assert parse_csv_line('a,') == ['a', '']",
   "assert parse_csv_line('\"\"') == ['']"],
  """
def parse_csv_line(line):
    out, cur, i, q = [], [], 0, False
    while i < len(line):
        c = line[i]
        if q:
            if c == '"':
                if i + 1 < len(line) and line[i+1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    q = False
            else:
                cur.append(c)
        else:
            if c == ',':
                out.append(''.join(cur))
                cur = []
            elif c == '"':
                q = True
            else:
                cur.append(c)
        i += 1
    out.append(''.join(cur))
    return out
""")

T("eval_rpn", "eval_rpn",
  """Write `eval_rpn(tokens: list[str]) -> int` evaluating integer Reverse Polish
Notation with + - * /. Division truncates toward zero (so -7/2 is -3).
Tokens may be negative numbers like '-3'.""",
  ["assert eval_rpn(['2','1','+','3','*']) == 9"],
  ["assert eval_rpn(['4','13','5','/','+']) == 6",
   "assert eval_rpn(['-7','2','/']) == -3",
   "assert eval_rpn(['7','-2','/']) == -3",
   "assert eval_rpn(['5']) == 5",
   "assert eval_rpn(['3','4','*','2','-']) == 10",
   "assert eval_rpn(['3','-4','-']) == 7"],
  """
def eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in ('+','-','*','/'):
            b = st.pop(); a = st.pop()
            if t == '+': st.append(a + b)
            elif t == '-': st.append(a - b)
            elif t == '*': st.append(a * b)
            else:
                q = abs(a) // abs(b)
                st.append(q if (a >= 0) == (b >= 0) else -q)
        else:
            st.append(int(t))
    return st[0]
""")

T("calculator", "calculate",
  """Write `calculate(expr: str) -> float` evaluating an arithmetic expression
with non-negative integer literals, + - * /, parentheses, unary minus and
arbitrary spaces. Usual precedence; / is true division; operators of equal
precedence are left-associative. Do not use eval, exec, compile or ast.""",
  ["assert calculate('1 + 2 * 3') == 7"],
  ["assert calculate('(1 + 2) * 3') == 9",
   "assert calculate('8 / 4 / 2') == 1",
   "assert calculate('10 - 4 - 3') == 3",
   "assert calculate('-(2 + 3) * 2') == -10",
   "assert calculate('2 * -3') == -6",
   "assert calculate(' 7 ') == 7",
   "assert calculate('1 / 4') == 0.25",
   "assert calculate('--3') == 3",
   "import re\nassert not re.search(r'\\b(eval|exec|compile)\\s*\\(|\\bimport\\s+ast|from\\s+ast', SOLUTION_SOURCE)"],
  """
def calculate(expr):
    s = expr.replace(' ', '')
    pos = 0
    def peek():
        return s[pos] if pos < len(s) else ''
    def eat():
        nonlocal pos
        pos += 1
        return s[pos-1]
    def expr_():
        v = term()
        while peek() in ('+', '-') and peek():
            op = eat()
            r = term()
            v = v + r if op == '+' else v - r
        return v
    def term():
        v = factor()
        while peek() in ('*', '/') and peek():
            op = eat()
            r = factor()
            v = v * r if op == '*' else v / r
        return v
    def factor():
        nonlocal pos
        c = peek()
        if c == '-':
            eat()
            return -factor()
        if c == '(':
            eat()
            v = expr_()
            eat()
            return v
        start = pos
        while peek().isdigit() and peek():
            pos += 1
        return int(s[start:pos])
    return expr_()
""")

T("next_permutation", "next_permutation",
  """Write `next_permutation(a: list) -> list` returning the next
lexicographically greater permutation of a as a new list. If a is the
greatest permutation, return the smallest (sorted ascending). Duplicates are
allowed.""",
  ["assert next_permutation([1,2,3]) == [1,3,2]"],
  ["assert next_permutation([3,2,1]) == [1,2,3]",
   "assert next_permutation([1,1,5]) == [1,5,1]",
   "assert next_permutation([1,5,1]) == [5,1,1]",
   "assert next_permutation([]) == []",
   "x=[1,2]\nnext_permutation(x)\nassert x == [1,2]"],
  """
def next_permutation(a):
    a = list(a)
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i+1]:
        i -= 1
    if i >= 0:
        j = len(a) - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
    a[i+1:] = reversed(a[i+1:])
    return a
""")

T("group_anagrams", "group_anagrams",
  """Write `group_anagrams(words: list[str]) -> list[list[str]]`. Group words
that are anagrams of each other. Sort each group alphabetically (keeping
duplicates) and sort the groups by their first element.""",
  ["assert group_anagrams(['ab','ba','c']) == [['ab','ba'],['c']]"],
  ["assert group_anagrams([]) == []",
   "assert group_anagrams(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]",
   "assert group_anagrams(['a','a']) == [['a','a']]",
   "assert group_anagrams(['','']) == [['','']]"],
  """
def group_anagrams(words):
    g = {}
    for w in words:
        g.setdefault(''.join(sorted(w)), []).append(w)
    return sorted((sorted(v) for v in g.values()), key=lambda v: v[0])
""")

T("max_subarray", "max_subarray",
  """Write `max_subarray(nums: list[int]) -> int` returning the largest sum of a
non-empty contiguous subarray. Raise ValueError on empty input.""",
  ["assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6"],
  ["assert max_subarray([-3,-1,-2]) == -1",
   "assert max_subarray([5]) == 5",
   "assert max_subarray([1,2,3]) == 6",
   "try:\n    max_subarray([])\n    raise AssertionError()\nexcept ValueError:\n    pass"],
  """
def max_subarray(nums):
    if not nums:
        raise ValueError('empty')
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
""")

T("coin_change", "coin_change",
  """Write `coin_change(coins: list[int], amount: int) -> int` returning the
minimum number of coins summing to amount (unlimited supply), or -1 if
impossible. amount 0 needs 0 coins.""",
  ["assert coin_change([1,2,5], 11) == 3"],
  ["assert coin_change([2], 3) == -1",
   "assert coin_change([1], 0) == 0",
   "assert coin_change([186,419,83,408], 6249) == 20",
   "assert coin_change([], 5) == -1"],
  """
def coin_change(coins, amount):
    INF = float('inf')
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a-c] + 1 < dp[a]:
                dp[a] = dp[a-c] + 1
    return -1 if dp[amount] == INF else dp[amount]
""")

T("primes_up_to", "primes_up_to",
  """Write `primes_up_to(n: int) -> list[int]` returning all primes <= n in
ascending order. Must handle n = 2,000,000 in about a second.""",
  ["assert primes_up_to(10) == [2,3,5,7]"],
  ["assert primes_up_to(1) == []",
   "assert primes_up_to(2) == [2]",
   "assert primes_up_to(25)[-1] == 23 and primes_up_to(49)[-1] == 47",
   "assert primes_up_to(-5) == []",
   "assert len(primes_up_to(2000000)) == 148933"],
  """
def primes_up_to(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(range(i*i, n + 1, i)))
        i += 1
    return [i for i, v in enumerate(sieve) if v]
""")

T("semver_compare", "semver_compare",
  """Write `semver_compare(a: str, b: str) -> int` returning -1, 0 or 1 comparing
Semantic Versioning 2.0.0 strings like '1.2.3', '1.0.0-alpha.1',
'1.0.0+build.5'. Follow the SemVer precedence rules: build metadata is
ignored; a pre-release version has lower precedence than the release;
pre-release identifiers are compared dot by dot, numeric identifiers
numerically, alphanumeric ones lexically in ASCII order, numeric identifiers
lower than alphanumeric ones, and a shorter set of identifiers is lower if
all preceding ones are equal.""",
  ["assert semver_compare('1.0.0', '2.0.0') == -1"],
  ["assert semver_compare('1.10.0', '1.9.0') == 1",
   "assert semver_compare('1.0.0-alpha', '1.0.0') == -1",
   "assert semver_compare('1.0.0+abc', '1.0.0+xyz') == 0",
   "chain = ['1.0.0-alpha','1.0.0-alpha.1','1.0.0-alpha.beta','1.0.0-beta','1.0.0-beta.2','1.0.0-beta.11','1.0.0-rc.1','1.0.0']\nfor x, y in zip(chain, chain[1:]):\n    assert semver_compare(x, y) == -1, (x, y)\n    assert semver_compare(y, x) == 1, (y, x)",
   "assert semver_compare('2.0.0', '2.0.0') == 0"
   ,"assert semver_compare('1.0.0-alpha.1', '1.0.0-alpha.1') == 0 and semver_compare('1.0.0-rc', '1.0.0-rc+b') == 0"],
  """
def semver_compare(a, b):
    def parse(v):
        v = v.split('+', 1)[0]
        core, _, pre = v.partition('-')
        nums = tuple(int(x) for x in core.split('.'))
        return nums, (pre.split('.') if pre else None)
    (na, pa), (nb, pb) = parse(a), parse(b)
    if na != nb:
        return -1 if na < nb else 1
    if pa is None and pb is None:
        return 0
    if pa is None:
        return 1
    if pb is None:
        return -1
    for x, y in zip(pa, pb):
        if x == y:
            continue
        xd, yd = x.isdigit(), y.isdigit()
        if xd and yd:
            return -1 if int(x) < int(y) else 1
        if xd:
            return -1
        if yd:
            return 1
        return -1 if x < y else 1
    if len(pa) == len(pb):
        return 0
    return -1 if len(pa) < len(pb) else 1
""")

T("deep_merge", "deep_merge",
  """Write `deep_merge(a: dict, b: dict) -> dict` returning a new dict that
recursively merges b into a: when both values for a key are dicts, merge
them; otherwise b's value wins. Neither input may be mutated, and the result
must not share nested dicts with the inputs.""",
  ["assert deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}"],
  ["assert deep_merge({'x': {'y': 1, 'z': 2}}, {'x': {'z': 3}}) == {'x': {'y': 1, 'z': 3}}",
   "assert deep_merge({'x': {'y': 1}}, {'x': 5}) == {'x': 5}",
   "a={'x':{'y':1}}; b={'x':{'z':2}}\nr=deep_merge(a,b)\nassert a == {'x':{'y':1}} and b == {'x':{'z':2}}",
   "a={'x':{'y':1}}\nr=deep_merge(a,{})\nr['x']['y']=99\nassert a['x']['y'] == 1",
   "b={'x':{'y':1}}\nr=deep_merge({},b)\nr['x']['y']=99\nassert b['x']['y'] == 1"],
  """
import copy
def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
""")

T("chunk", "chunk",
  """Write `chunk(iterable, n: int) -> list[list]` splitting any iterable
(including generators) into consecutive lists of size n; the last may be
shorter. Raise ValueError if n < 1.""",
  ["assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]"],
  ["assert chunk([], 3) == []",
   "assert chunk((x for x in range(4)), 2) == [[0,1],[2,3]]",
   "assert chunk('abc', 5) == [['a','b','c']]",
   "assert chunk([1,2,3], 1) == [[1],[2],[3]]",
   "try:\n    chunk([1], 0)\n    raise AssertionError()\nexcept ValueError:\n    pass"],
  """
from itertools import islice
def chunk(iterable, n):
    if n < 1:
        raise ValueError(n)
    it = iter(iterable)
    out = []
    while True:
        c = list(islice(it, n))
        if not c:
            return out
        out.append(c)
""")

T("humanize_bytes", "humanize_bytes",
  """Write `humanize_bytes(n: int) -> str`. Below 1024 return e.g. '512 B'.
Otherwise divide by 1024 repeatedly and use units KiB, MiB, GiB, TiB, PiB
(never beyond PiB), formatted with exactly one decimal: '1.5 KiB'. Pick the
largest unit where the value is >= 1. Negative n raises ValueError.""",
  ["assert humanize_bytes(1536) == '1.5 KiB'"],
  ["assert humanize_bytes(0) == '0 B'",
   "assert humanize_bytes(1023) == '1023 B'",
   "assert humanize_bytes(1024) == '1.0 KiB'",
   "assert humanize_bytes(1024**2) == '1.0 MiB' and humanize_bytes(1024**2 - 1) == '1024.0 KiB'",
   "assert humanize_bytes(1024**3 * 5) == '5.0 GiB'",
   "assert humanize_bytes(1024**6) == '1024.0 PiB'",
   "try:\n    humanize_bytes(-1)\n    raise AssertionError()\nexcept ValueError:\n    pass"],
  """
def humanize_bytes(n):
    if n < 0:
        raise ValueError(n)
    if n < 1024:
        return f'{n} B'
    v = float(n)
    for u in ['KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        v /= 1024
        if v < 1024 or u == 'PiB':
            return f'{v:.1f} {u}'
""")

T("parse_query", "parse_query",
  """Write `parse_query(q: str) -> dict[str, list[str]]` parsing a URL query
string (without the leading '?'). Pairs are separated by '&'; a key without
'=' has value ''. '+' means space, and %XX escapes are percent-decoded as
UTF-8, in both keys and values. Repeated keys collect values in order. Empty
segments are skipped. Do not use urllib.""",
  ["assert parse_query('a=1&b=2&a=3') == {'a': ['1', '3'], 'b': ['2']}"],
  ["assert parse_query('') == {}",
   "assert parse_query('x') == {'x': ['']}",
   "assert parse_query('q=hello+world') == {'q': ['hello world']}",
   "assert parse_query('n=%C3%A9t%C3%A9&&k%20y=a%3Db') == {'n': ['été'], 'k y': ['a=b']}",
   "assert parse_query('a=1=2') == {'a': ['1=2']}"],
  """
def parse_query(q):
    def dec(s):
        s = s.replace('+', ' ')
        out = bytearray()
        i = 0
        while i < len(s):
            if s[i] == '%' and i + 2 < len(s) + 0 and len(s[i+1:i+3]) == 2:
                out += bytes([int(s[i+1:i+3], 16)])
                i += 3
            else:
                out += s[i].encode()
                i += 1
        return out.decode('utf-8')
    res = {}
    for seg in q.split('&'):
        if not seg:
            continue
        k, _, v = seg.partition('=')
        res.setdefault(dec(k), []).append(dec(v))
    return res
""")

T("longest_palindrome", "longest_palindrome",
  """Write `longest_palindrome(s: str) -> str` returning the longest palindromic
substring; on ties return the one that starts earliest. Should run in O(n^2)
or better for n = 3000.""",
  ["assert longest_palindrome('babad') == 'bab'"],
  ["assert longest_palindrome('') == ''",
   "assert longest_palindrome('cbbd') == 'bb'",
   "assert longest_palindrome('abc') == 'a'",
   "assert longest_palindrome('forgeeksskeegfor') == 'geeksskeeg'",
   "assert longest_palindrome('ab' * 1500) == ('ab' * 1500)[:2999]"],
  """
def longest_palindrome(s):
    best = (0, 0)
    for c in range(len(s)):
        for l, r in ((c, c), (c, c + 1)):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1; r += 1
            if r - l - 1 > best[1] - best[0]:
                best = (l + 1, r)
    return s[best[0]:best[1]]
""")

T("count_islands", "count_islands",
  """Write `count_islands(grid: list[str]) -> int`. grid is a list of equal
length strings of '1' (land) and '0' (water). Count groups of land connected
horizontally or vertically. Grids can be 300x300; avoid recursion depth
errors.""",
  ["assert count_islands(['110', '010', '001']) == 2"],
  ["assert count_islands([]) == 0",
   "assert count_islands(['000']) == 0",
   "assert count_islands(['101', '010', '101']) == 5",
   "assert count_islands(['1' * 300] * 300) == 1",
   "g = [''.join('1' if (r % 2 == 0 or c == (0 if (r // 2) % 2 else 299)) else '0' for c in range(300)) for r in range(300)]\nassert count_islands(g) == 1"],
  """
def count_islands(grid):
    if not grid:
        return 0
    R, C = len(grid), len(grid[0])
    seen = [[False] * C for _ in range(R)]
    n = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '1' and not seen[r][c]:
                n += 1
                st = [(r, c)]
                seen[r][c] = True
                while st:
                    y, x = st.pop()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < R and 0 <= nx < C and grid[ny][nx] == '1' and not seen[ny][nx]:
                            seen[ny][nx] = True
                            st.append((ny, nx))
    return n
""")

T("min_window", "min_window",
  """Write `min_window(s: str, t: str) -> str` returning the shortest substring
of s containing every character of t (with multiplicity). On ties return the
leftmost. Return '' if none exists or t is empty.""",
  ["assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'"],
  ["assert min_window('a', 'aa') == ''",
   "assert min_window('aa', 'aa') == 'aa'",
   "assert min_window('abc', '') == ''",
   "assert min_window('abcabc', 'cb') == 'bc'",
   "assert min_window('bba', 'ab') == 'ba'",
   "assert min_window('aaflslflsldkalskaaa', 'aaa') == 'aaa'",
   "assert min_window('cabwefgewcwaefgcf', 'cae') == 'cwae'",
   "assert min_window('x' * 20000 + 'ab', 'ba') == 'ab'"],
  """
from collections import Counter
def min_window(s, t):
    if not t:
        return ''
    need = Counter(t)
    missing = len(t)
    best = (0, float('inf'))
    l = 0
    for r, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        if missing == 0:
            while need[s[l]] < 0:
                need[s[l]] += 1
                l += 1
            if r - l < best[1] - best[0]:
                best = (l, r)
            need[s[l]] += 1
            missing += 1
            l += 1
    return '' if best[1] == float('inf') else s[best[0]:best[1] + 1]
""")

T("trie", "Trie",
  """Write a class `Trie` with `insert(word)`, `search(word) -> bool` (exact
word was inserted) and `starts_with(prefix) -> bool` (some inserted word has
this prefix; the empty prefix is True only if at least one word was
inserted).""",
  ["t = Trie()\nt.insert('apple')\nassert t.search('apple')"],
  ["t = Trie()\nt.insert('apple')\nassert not t.search('app')\nassert t.starts_with('app')\nt.insert('app')\nassert t.search('app')",
   "t = Trie()\nassert not t.starts_with('')\nt.insert('')\nassert t.search('')\nassert t.starts_with('')",
   "t = Trie()\nt.insert('ab')\nassert not t.starts_with('abc')\nassert not t.search('a')"],
  """
class Trie:
    def __init__(self):
        self.root = {}
        self.count = 0
    def insert(self, word):
        n = self.root
        for c in word:
            n = n.setdefault(c, {})
        n['$'] = True
        self.count += 1
    def _find(self, s):
        n = self.root
        for c in s:
            if c not in n:
                return None
            n = n[c]
        return n
    def search(self, word):
        n = self._find(word)
        return bool(n) and n.get('$', False)
    def starts_with(self, prefix):
        if prefix == '':
            return self.count > 0
        return self._find(prefix) is not None
""")

T("two_sum", "two_sum",
  """Write `two_sum(nums: list[int], target: int)` returning a tuple (i, j) with
i < j and nums[i] + nums[j] == target, choosing the pair with the smallest j,
and for that j the smallest i. Return None if no pair exists. Must be O(n).""",
  ["assert two_sum([2,7,11,15], 9) == (0, 1)"],
  ["assert two_sum([3,3,3], 6) == (0, 1)",
   "assert two_sum([1,2,3,4], 7) == (2, 3)",
   "assert two_sum([1,5,1,5], 6) == (0, 1)",
   "assert two_sum([1], 2) is None",
   "assert two_sum([], 0) is None",
   "import time\nt=time.time()\nassert two_sum(list(range(200000)), 399997) == (199998, 199999)\nassert time.time()-t < 1"],
  """
def two_sum(nums, target):
    first = {}
    for j, x in enumerate(nums):
        if target - x in first:
            return (first[target - x], j)
        first.setdefault(x, j)
    return None
""")

T("moving_average", "moving_average",
  """Write `moving_average(nums: list[float], k: int) -> list[float]` returning
the mean of each window of k consecutive values (len(nums) - k + 1 results).
Raise ValueError if k < 1 or k > len(nums). Must be O(n).""",
  ["assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]"],
  ["assert moving_average([5], 1) == [5]",
   "assert moving_average([1, 2, 3], 3) == [2]",
   "for k in (0, 4):\n    try:\n        moving_average([1,2,3], k)\n        raise AssertionError(k)\n    except ValueError:\n        pass",
   "r = moving_average(list(range(200000)), 1000)\nassert len(r) == 199001 and abs(r[0] - 499.5) < 1e-9 and abs(r[-1] - 199499.5) < 1e-6"],
  """
def moving_average(nums, k):
    if k < 1 or k > len(nums):
        raise ValueError(k)
    s = sum(nums[:k])
    out = [s / k]
    for i in range(k, len(nums)):
        s += nums[i] - nums[i-k]
        out.append(s / k)
    return out
""")

T("compress_ranges", "compress_ranges",
  """Write `compress_ranges(nums: list[int]) -> str` that sorts and de-duplicates
nums and renders consecutive runs of 3 or more as 'a-b'; runs of 1 or 2 are
listed individually. Items are joined by ','. Negative numbers are written
plainly, e.g. '-3--1'.""",
  ["assert compress_ranges([1,2,3,5,7,8]) == '1-3,5,7,8'"],
  ["assert compress_ranges([]) == ''",
   "assert compress_ranges([3,1,2,2]) == '1-3'",
   "assert compress_ranges([-3,-2,-1,1]) == '-3--1,1'",
   "assert compress_ranges([10]) == '10'",
   "assert compress_ranges([1,2,4,5,6,7]) == '1,2,4-7'"],
  """
def compress_ranges(nums):
    xs = sorted(set(nums))
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j+1] == xs[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f'{xs[i]}-{xs[j]}')
        else:
            out.extend(str(x) for x in xs[i:j+1])
        i = j + 1
    return ','.join(out)
""")

T("expand_ranges", "expand_ranges",
  """Write `expand_ranges(s: str) -> list[int]` expanding strings like
'1-3, 5,7-8' into [1, 2, 3, 5, 7, 8], preserving the given order and
duplicates. Items are non-negative integers or a-b with a <= b; spaces around
items are allowed. Raise ValueError for a > b or malformed items. Empty or
whitespace-only input gives [].""",
  ["assert expand_ranges('1-3,5') == [1, 2, 3, 5]"],
  ["assert expand_ranges('') == []",
   "assert expand_ranges('  ') == []",
   "assert expand_ranges(' 7 , 2-3 ') == [7, 2, 3]",
   "assert expand_ranges('4-4') == [4]",
   "for bad in ['3-1', 'a', '1-', '1,,2', '1-2-3']:\n    try:\n        expand_ranges(bad)\n        raise AssertionError(bad)\n    except ValueError:\n        pass"],
  """
import re
def expand_ranges(s):
    if not s.strip():
        return []
    out = []
    for item in s.split(','):
        m = re.fullmatch(r'\\s*(\\d+)(?:-(\\d+))?\\s*', item)
        if not m:
            raise ValueError(item)
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if a > b:
            raise ValueError(item)
        out.extend(range(a, b + 1))
    return out
""")

T("sudoku_valid", "is_valid_sudoku",
  """Write `is_valid_sudoku(board: list[str]) -> bool` for a 9x9 board given as 9
strings of 9 characters, digits '1'-'9' or '.' for empty. Return True if no
row, column or 3x3 box contains a repeated digit (the board need not be
solvable or complete).""",
  ["assert is_valid_sudoku(['.' * 9] * 9) is True"],
  ["b = ['53..7....','6..195...','.98....6.','8...6...3','4..8.3..1','7...2...6','.6....28.','...419..5','....8..79']\nassert is_valid_sudoku(b) is True",
   "b = ['83..7....','6..195...','.98....6.','8...6...3','4..8.3..1','7...2...6','.6....28.','...419..5','....8..79']\nassert is_valid_sudoku(b) is False",
   "b = ['1........','.1.......'] + ['.'*9]*7\nassert is_valid_sudoku(b) is False",
   "b = ['1'+'.'*8] + ['.'*9]*7 + ['.'*8+'1']\nassert is_valid_sudoku(b) is True"],
  """
def is_valid_sudoku(board):
    seen = set()
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == '.':
                continue
            keys = [('r', r, v), ('c', c, v), ('b', r // 3, c // 3, v)]
            if any(k in seen for k in keys):
                return False
            seen.update(keys)
    return True
""")

T("kth_largest", "kth_largest",
  """Write `kth_largest(nums: list[int], k: int) -> int` returning the k-th
largest element (1-based, duplicates count). Raise ValueError if k is out of
range. Do not mutate nums.""",
  ["assert kth_largest([3,2,1,5,6,4], 2) == 5"],
  ["assert kth_largest([3,2,3,1,2,4,5,5,6], 4) == 4",
   "assert kth_largest([1], 1) == 1",
   "x=[2,1]\nkth_largest(x,1)\nassert x == [2,1]",
   "for k in (0, 3):\n    try:\n        kth_largest([1,2], k)\n        raise AssertionError(k)\n    except ValueError:\n        pass"],
  """
import heapq
def kth_largest(nums, k):
    if not 1 <= k <= len(nums):
        raise ValueError(k)
    return heapq.nlargest(k, nums)[-1]
""")

T("token_bucket", "TokenBucket",
  """Write a class `TokenBucket(capacity: float, rate: float)`. It starts full.
Tokens refill continuously at `rate` per second up to `capacity`.
`allow(now: float, cost: float = 1) -> bool` refills based on the time since
the previous call (the first call counts from the time of that first call),
then if at least `cost` tokens are available, consumes them and returns True;
otherwise returns False and consumes nothing. Time never goes backwards.""",
  ["b = TokenBucket(2, 1)\nassert b.allow(0) and b.allow(0) and not b.allow(0)"],
  ["b = TokenBucket(2, 1)\nassert b.allow(100) and b.allow(100)\nassert not b.allow(100.5)\nassert b.allow(101)",
   "b = TokenBucket(1, 10)\nassert b.allow(0)\nassert not b.allow(0.05)\nassert b.allow(0.1)",
   "b = TokenBucket(3, 1)\nassert not b.allow(0, cost=4)\nassert b.allow(0, cost=3)",
   "b = TokenBucket(2, 1)\nassert b.allow(0, 2)\nassert not b.allow(1000, 3)\nassert b.allow(1000, 2)"],
  """
class TokenBucket:
    def __init__(self, capacity, rate):
        self.cap = capacity
        self.rate = rate
        self.tokens = capacity
        self.last = None
    def allow(self, now, cost=1):
        if self.last is not None:
            self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens + 1e-12 >= cost:
            self.tokens -= cost
            return True
        return False
""")

T("retry", "retry",
  """Write a decorator factory `retry(times: int, exceptions=(Exception,))`.
The decorated function is called up to `times` times in total; if it raises
one of `exceptions`, it is retried; any other exception propagates
immediately. If all attempts fail, re-raise the last exception. Preserve the
wrapped function's __name__ and __doc__. times < 1 raises ValueError when
retry() is called.""",
  ["calls = []\n@retry(3)\ndef f():\n    calls.append(1)\n    return 'ok'\nassert f() == 'ok' and len(calls) == 1"],
  ["n = [0]\n@retry(3)\ndef f(x):\n    n[0] += 1\n    if n[0] < 3: raise IOError('x')\n    return x * 2\nassert f(4) == 8 and n[0] == 3",
   "n = [0]\n@retry(2, exceptions=(KeyError,))\ndef f():\n    n[0] += 1\n    raise KeyError(n[0])\ntry:\n    f()\n    raise AssertionError()\nexcept KeyError as e:\n    assert e.args == (2,)",
   "n = [0]\n@retry(5, exceptions=(KeyError,))\ndef f():\n    n[0] += 1\n    raise TypeError()\ntry:\n    f()\nexcept TypeError:\n    pass\nassert n[0] == 1",
   "@retry(1)\ndef g():\n    'doc'\nassert g.__name__ == 'g' and g.__doc__ == 'doc'",
   "try:\n    retry(0)\n    raise AssertionError()\nexcept ValueError:\n    pass"],
  """
import functools
def retry(times, exceptions=(Exception,)):
    if times < 1:
        raise ValueError(times)
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            last = None
            for _ in range(times):
                try:
                    return fn(*a, **kw)
                except exceptions as e:
                    last = e
            raise last
        return wrapper
    return deco
""")

T("json_path_get", "get_path",
  """Write `get_path(obj, path: str, default=None)`. path uses dot keys and
[index] for lists, e.g. 'a.b[0].c' or '[1].x' or 'a[0][2]'. Return the value,
or default if any step is missing, out of range, or the wrong type. Negative
indexes are allowed like Python. An empty path returns obj.""",
  ["assert get_path({'a': {'b': [{'c': 5}]}}, 'a.b[0].c') == 5"],
  ["assert get_path({'a': 1}, 'b', 'd') == 'd'",
   "assert get_path([{'x': 1}, {'x': 2}], '[1].x') == 2",
   "assert get_path({'a': [[1,2,3]]}, 'a[0][2]') == 3",
   "assert get_path({'a': [1]}, 'a[5]') is None",
   "assert get_path({'a': [1, 2]}, 'a[-1]') == 2",
   "assert get_path({'a': [1, 2]}, 'a[2]') is None and get_path({'a': [1, 2]}, 'a[-3]') is None and get_path({'a': [1, 2]}, 'a[-2]') == 1",
   "assert get_path({'a': 'str'}, 'a.b') is None",
   "assert get_path({'a': 'str'}, 'a[0]') is None",
   "assert get_path({'a': 0}, 'a', 9) == 0",
   "assert get_path({'k': 1}, '') == {'k': 1}"],
  """
import re
def get_path(obj, path, default=None):
    cur = obj
    for key, idx in re.findall(r'([^.\\[\\]]+)|\\[(-?\\d+)\\]', path):
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        else:
            i = int(idx)
            if not isinstance(cur, list) or not -len(cur) <= i < len(cur):
                return default
            cur = cur[i]
    return cur
""")

T("max_meetings", "max_meetings",
  """Write `max_meetings(intervals: list[tuple[int, int]]) -> int` returning the
maximum number of non-overlapping meetings one person can attend. Each
meeting is (start, end) with start < end; a meeting may start exactly when
another ends.""",
  ["assert max_meetings([(1, 2), (2, 3), (1, 3)]) == 2"],
  ["assert max_meetings([]) == 0",
   "assert max_meetings([(1, 10), (2, 3), (3, 4), (4, 5)]) == 3",
   "assert max_meetings([(1, 5), (1, 5), (1, 5)]) == 1",
   "assert max_meetings([(5, 6), (1, 2), (3, 4)]) == 3"],
  """
def max_meetings(intervals):
    n, end = 0, float('-inf')
    for s, e in sorted(intervals, key=lambda x: x[1]):
        if s >= end:
            n += 1
            end = e
    return n
""")

T("format_table", "format_table",
  """Write `format_table(rows: list[list]) -> str`. The first row is the header.
Convert every cell with str(). Pad each column to its widest cell with
left-aligned text, separate columns with ' | ', and after the header insert a
line of dashes where each column is dashes of its width, joined by '-+-'.
Lines are joined with '\\n' with no trailing newline and no trailing spaces on
any line. Empty rows input returns ''.""",
  ["assert format_table([['a']]) == 'a\\n-'"],
  ["assert format_table([]) == ''",
   "assert format_table([['name','age'],['bob',7]]) == 'name | age\\n-----+----\\nbob  | 7'",
   "assert format_table([['x','y'],['longer','z']]) == 'x      | y\\n-------+--\\nlonger | z'",
   "assert format_table([['a','b'],[1,'']]) == 'a | b\\n--+--\\n1 |'"],
  """
def format_table(rows):
    if not rows:
        return ''
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    def line(r):
        return ' | '.join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip()
    out = [line(rows[0]), '-+-'.join('-' * x for x in w)]
    out += [line(r) for r in rows[1:]]
    return '\\n'.join(out)
""")

T("normalize_path", "normalize_path",
  """Write `normalize_path(p: str) -> str` normalizing a POSIX path without
touching the filesystem. Collapse repeated slashes, drop '.', resolve '..'.
For absolute paths, '..' at the root stays at the root. For relative paths,
leading '..' segments that cannot be resolved are kept. A trailing slash is
dropped (except for '/'). An empty or fully-cancelled relative path becomes
'.'. Do not use os.path or pathlib.""",
  ["assert normalize_path('/a/./b/../c/') == '/a/c'"],
  ["assert normalize_path('/../..') == '/'",
   "assert normalize_path('a/../../b') == '../b'",
   "assert normalize_path('') == '.'",
   "assert normalize_path('a/..') == '.'",
   "assert normalize_path('//x///y') == '/x/y'",
   "assert normalize_path('../../a') == '../../a'",
   "assert normalize_path('./') == '.'"],
  """
def normalize_path(p):
    absolute = p.startswith('/')
    out = []
    for seg in p.split('/'):
        if seg in ('', '.'):
            continue
        if seg == '..':
            if out and out[-1] != '..':
                out.pop()
            elif not absolute:
                out.append('..')
        else:
            out.append(seg)
    s = '/'.join(out)
    if absolute:
        return '/' + s
    return s or '.'
""")

T("bank_transfers", "settle",
  """Write `settle(balances: dict[str, int]) -> list[tuple[str, str, int]]`.
balances maps person -> net amount (positive: is owed money, negative: owes
money); values sum to zero, otherwise raise ValueError. Return transfers
(payer, payee, amount) that settle all debts, produced greedily: repeatedly
take the person who owes the most and the person owed the most (ties broken
alphabetically by name) and transfer the smaller of the two magnitudes.
People with zero balance never appear.""",
  ["assert settle({'a': -5, 'b': 5}) == [('a', 'b', 5)]"],
  ["assert settle({}) == []",
   "assert settle({'a': 0}) == []",
   "assert settle({'a': -10, 'b': 4, 'c': 6}) == [('a', 'c', 6), ('a', 'b', 4)]",
   "assert settle({'x': -3, 'y': -3, 'z': 6}) == [('x', 'z', 3), ('y', 'z', 3)]",
   "try:\n    settle({'a': 1})\n    raise AssertionError()\nexcept ValueError:\n    pass"],
  """
def settle(balances):
    if sum(balances.values()) != 0:
        raise ValueError('unbalanced')
    bal = {k: v for k, v in balances.items() if v}
    out = []
    while bal:
        payer = min(bal, key=lambda k: (bal[k], k))
        payee = min(bal, key=lambda k: (-bal[k], k))
        amt = min(-bal[payer], bal[payee])
        out.append((payer, payee, amt))
        bal[payer] += amt
        bal[payee] -= amt
        bal = {k: v for k, v in bal.items() if v}
    return out
""")

T("interleave_iters", "roundrobin",
  """Write `roundrobin(*iterables)` as a generator that yields one item from each
iterable in turn, skipping exhausted ones, until all are exhausted. It must
work lazily with infinite iterators (take only what is consumed).""",
  ["assert list(roundrobin('ABC', 'D', 'EF')) == ['A', 'D', 'E', 'B', 'F', 'C']"],
  ["assert list(roundrobin()) == []",
   "assert list(roundrobin([], [1])) == [1]",
   "import itertools\ng = roundrobin(itertools.count(), 'ab')\nassert list(itertools.islice(g, 6)) == [0, 'a', 1, 'b', 2, 3]",
   "import types\nassert isinstance(roundrobin([1]), types.GeneratorType)"],
  """
def roundrobin(*iterables):
    its = [iter(i) for i in iterables]
    while its:
        nxt = []
        for it in its:
            try:
                yield next(it)
            except StopIteration:
                continue
            nxt.append(it)
        its = nxt
""")

T("balanced_parens_gen", "generate_parens",
  """Write `generate_parens(n: int) -> list[str]` returning all strings of n
pairs of well-formed parentheses in lexicographic order ('(' < ')').
n = 0 gives [''].""",
  ["assert generate_parens(2) == ['(())', '()()']"],
  ["assert generate_parens(0) == ['']",
   "assert generate_parens(1) == ['()']",
   "assert generate_parens(3) == ['((()))', '(()())', '(())()', '()(())', '()()()']",
   "assert len(generate_parens(8)) == 1430"],
  """
def generate_parens(n):
    out = []
    def go(s, o, c):
        if len(s) == 2 * n:
            out.append(s)
            return
        if o < n:
            go(s + '(', o + 1, c)
        if c < o:
            go(s + ')', o, c + 1)
    go('', 0, 0)
    return out
""")

T("diff_dicts", "diff_dicts",
  """Write `diff_dicts(old: dict, new: dict) -> dict` returning
{'added': {...}, 'removed': {...}, 'changed': {key: (old_value, new_value)}}
comparing top-level keys only. Values are compared with ==, but note that 1
and True, or 0 and False, must count as changed (their types differ).""",
  ["assert diff_dicts({'a': 1}, {'a': 2}) == {'added': {}, 'removed': {}, 'changed': {'a': (1, 2)}}"],
  ["assert diff_dicts({}, {}) == {'added': {}, 'removed': {}, 'changed': {}}",
   "assert diff_dicts({'a': 1, 'b': 2}, {'b': 2, 'c': 3}) == {'added': {'c': 3}, 'removed': {'a': 1}, 'changed': {}}",
   "r = diff_dicts({'a': 1, 'b': 0}, {'a': True, 'b': False})\nassert set(r['changed']) == {'a', 'b'}",
   "assert diff_dicts({'a': [1]}, {'a': [1]})['changed'] == {}"],
  """
def diff_dicts(old, new):
    added = {k: new[k] for k in new if k not in old}
    removed = {k: old[k] for k in old if k not in new}
    changed = {}
    for k in old:
        if k in new:
            a, b = old[k], new[k]
            if type(a) is not type(b) or a != b:
                changed[k] = (a, b)
    return {'added': added, 'removed': removed, 'changed': changed}
""")


def _split(i):
    return "heldout" if i % 5 in (1, 3) else "train"


TASKS = [Task(id=r[0], entry=r[1], prompt=r[2], examples=r[3], tests=r[4], reference=r[5], split=_split(i))
         for i, r in enumerate(_RAW)]


def get_tasks(split=None):
    if split in (None, "all"):
        return list(TASKS)
    return [t for t in TASKS if t.split == split]
