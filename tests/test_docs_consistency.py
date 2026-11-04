"""Docs vs code consistency checks (stdlib only, reads files, nothing else).

These tests fail when the documentation drifts from the code: a route, an event,
a file or a CI step that exists in the repo but is missing from the docs (or the
other way round), a broken README anchor, or a stale name that was removed.
When one fails after you add a route / file / event, update the docs it names.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def _headings(text):
    """(level, title) for every markdown heading outside fenced code blocks."""
    out = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            out.append((len(m.group(1)), m.group(2)))
    return out


def _slug(title):
    """GitHub heading anchor rule (without the duplicate suffix)."""
    t = title.replace("**", "").replace("`", "")
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def _slugs(text):
    seen = {}
    out = []
    for _level, title in _headings(text):
        s = _slug(title)
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.append(s if n == 0 else f"{s}-{n}")
    return out


def _fenced_block_after(text, heading_fragment):
    """First fenced code block that follows the heading containing heading_fragment."""
    start = text.index(heading_fragment)
    m = re.search(r"```[^\n]*\n(.*?)\n```", text[start:], re.S)
    assert m, f"no code block after {heading_fragment!r}"
    return m.group(1)


def _section(text, heading_fragment):
    """Text from the heading containing heading_fragment up to the next '---' rule."""
    start = text.index(heading_fragment)
    end = text.find("\n---\n", start)
    return text[start:end if end != -1 else len(text)]


def _root_py():
    return sorted(p for p in ROOT.glob("*.py"))


# ---------------------------------------------------------------- slug rule

def test_slug_rule_is_pinned():
    assert _slug("🌟 Key Features") == "-key-features"
    assert _slug("Job Order & Preview Pass (Run Settings)") == "job-order--preview-pass-run-settings"
    assert _slug("Distribution API (registered via `distribution_routes.py`)") == \
        "distribution-api-registered-via-distribution_routespy"
    assert _slug("[Link](#x) **Bold**") == "link-bold"


# ---------------------------------------------------------------- README

def test_readme_anchor_links_resolve():
    text = _read("README.md")
    assert text.count("```") % 2 == 0, "unbalanced code fences in README.md"
    slugs = set(_slugs(text))
    # Raw HTML anchors (name= / id=) also count as targets.
    slugs |= set(re.findall(r'(?:name|id)="([^"]+)"', text))
    scan = re.sub(r"```.*?```", "", text, flags=re.S)
    missing = sorted({a for a in re.findall(r"\]\(#([^)\s]+)\)", scan) if a not in slugs})
    assert not missing, f"README anchors with no heading: {missing}"


def test_readme_toc_covers_sections():
    text = _read("README.md")
    toc = _section(text, "## 📑 Table of Contents")
    linked = set(re.findall(r"\]\(#([^)\s]+)\)", toc))
    titles = _headings(text)
    skip = {titles[0][1], "📑 Table of Contents"}
    missing = [t for lvl, t in titles if lvl == 2 and t not in skip and _slug(t) not in linked]
    assert not missing, f"README '## ' sections absent from the TOC: {missing}"
    required = [
        "Workflow Sweep", "Pairwise Review", "Standalone Dashboard", "Image Metrics",
        "Job Order", "Skip Layer Guidance", "MOSS", "Run API", "Development / Tests",
    ]
    by_title = {t: _slug(t) for _lvl, t in titles}
    for frag in required:
        hits = [s for t, s in by_title.items() if frag in t]
        assert hits, f"README has no heading containing {frag!r}"
        assert any(s in linked for s in hits), f"TOC does not link {frag!r}"


def test_changelog_has_11_4():
    text = _read("README.md")
    start = text.index("## 📝 Changelog")
    first = re.search(r"^### (Update .*)$", text[start:], re.M)
    assert first and first.group(1).startswith("Update 11/4/26"), \
        f"newest changelog entry is {first.group(1) if first else None!r}"


# ---------------------------------------------------------------- ProjectStructure

def test_routes_documented():
    doc = _read("ProjectStructure.md").split("\n")
    pat = re.compile(r'routes\.(get|post|put|delete)\(\s*"([^"]+)"')
    missing = []
    for p in _root_py():
        if p.name == "conftest.py":
            continue
        for method, path in pat.findall(p.read_text(encoding="utf-8")):
            needle = "`" + path + "`"
            if not any(needle in line and method.upper() in line for line in doc):
                missing.append(f"{method.upper()} {path} ({p.name})")
    assert not missing, "routes missing from ProjectStructure.md API Routes: " + ", ".join(missing)


def test_file_map_complete():
    text = _read("ProjectStructure.md")
    block = _fenced_block_after(text, "Project File Structure")
    names = [p.name for p in _root_py()]
    names += [p.name for p in (ROOT / "web").rglob("*") if p.suffix in (".js", ".md")]
    names += [p.name for p in (ROOT / "resources").glob("*") if p.is_file()]
    names += ["tests/e2e/", "tests/js/", "examples/", "tests.yml", "pytest.ini"]
    missing = sorted({n for n in names if n not in block})
    assert not missing, f"files absent from the ProjectStructure.md file map: {missing}"
    on_disk = {p.name for p in ROOT.rglob("*.py") if ".git" not in p.parts}
    ghosts = sorted({n for n in re.findall(r"[\w\-]+\.py\b", block) if n not in on_disk})
    assert not ghosts, f"file map names .py files that do not exist: {ghosts}"


def test_events_documented():
    pat = re.compile(r'send_sync\(\s*"(ultimate_grid\.\w+)"')
    sent = set()
    for p in _root_py():
        sent |= set(pat.findall(p.read_text(encoding="utf-8")))
    assert sent, "no events found in the code"
    text = _read("ProjectStructure.md")
    sec = _section(text, "### **Server Event Types**")
    missing = sorted(e for e in sent if "`" + e + "`" not in sec)
    assert not missing, f"events missing from Server Event Types: {missing}"
    assert "phase" in sec and '"sweep"' in sec, "progress payload must document phase and the sweep payload"


def test_js_files_order():
    src = _read("html_generator.py")
    m = re.search(r"js_files\s*=\s*\[(.*?)\n\s*\]", src, re.S)
    assert m
    files = re.findall(r'^\s*"([\w\-]+\.js)"', m.group(1), re.M)
    assert files and files[-1] == "logic_standalone.js"
    for f in files:
        assert (ROOT / "resources" / f).is_file(), f
    assert files.index("logic_ratings.js") < files.index("logic_ui.js")
    assert files.index("logic_pipeline.js") < files.index("logic_compare.js")
    for name in ("ProjectStructure.md", "AI-CONTEXT.md"):
        doc = _read(name)
        pos = [doc.find(f) for f in files]
        assert all(p != -1 for p in pos), f"{name} must list every js_files entry"


def test_no_stale_names():
    for name in ("AI-CONTEXT.md", "ProjectStructure.md"):
        for no, line in enumerate(_read(name).split("\n"), 1):
            for stale in ("convertStateToConfigs", "network_utils.py"):
                if stale in line and "no longer" not in line and "removed" not in line:
                    raise AssertionError(f"{name}:{no} mentions {stale}")


# ---------------------------------------------------------------- CONTRIBUTING vs CI

def test_contributing_matches_ci():
    ci = _read(".github/workflows/tests.yml")
    jobs = ci[ci.index("\njobs:"):]
    ids = re.findall(r"^  ([\w\-]+):\s*$", jobs, re.M)
    assert ids, "no job ids found in tests.yml"
    contrib = _read("CONTRIBUTING.md")
    missing = [i for i in ids if i not in contrib]
    assert not missing, f"CI jobs not named in CONTRIBUTING.md: {missing}"
    for flag in ("--ignore=tests/e2e", "--browser chromium", "node --test tests/js/", "--input-type=module"):
        assert flag in contrib, f"CONTRIBUTING.md does not mention {flag!r}"
