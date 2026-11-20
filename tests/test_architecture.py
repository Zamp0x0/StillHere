"""Enforce the module dependency rules in docs/ARCHITECTURE.md so parallel work cannot tangle the layers."""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "stillhere"

# module -> stillhere modules it may import (besides itself)
ALLOWED = {
    "contracts": set(),
    "data": {"contracts"},
    "memory": {"contracts"},
    "inference": {"contracts", "data"},
    "importers": {"contracts", "memory"},
    "evaluation": {"contracts", "context", "memory", "observation", "retrieval", "writer"},
    "context": {"contracts", "memory"},
    "writer": {"contracts", "context", "memory"},
    "observation": {"contracts", "context", "memory"},
    "retrieval": {"contracts", "memory"},
    "style": {"contracts"},
    "style_eval": {"style"},
    "runtime": {"contracts", "memory", "style"},
    "session": {"contracts", "context", "memory", "runtime", "style"},     # Simple Context v1: live sessions over the existing chain
    "reader": {"contracts", "context", "session"},                          # EF1 fact reader: prompts over the fact pack (+ the pinned D4 renderer)
    "providers": set(),                                                     # per-role LLM providers (DeepSeek / custom / local): standard library only
    "persona": {"contracts", "context", "memory", "providers", "reader", "retrieval", "runtime", "session", "style", "style_eval", "writer"},   # persona pipeline: data, card (style profiler), memory (EF1), Fact Reader B, eval
}


def stillhere_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "stillhere" and len(parts) > 1:
                yield parts[1]


def test_every_module_is_declared():
    modules = {p.name for p in PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert modules == set(ALLOWED)


def test_dependency_rules():
    violations = []
    for module, allowed in ALLOWED.items():
        for path in (PACKAGE / module).rglob("*.py"):
            for target in stillhere_imports(path):
                if target != module and target not in allowed:
                    violations.append(f"{path.relative_to(PACKAGE.parent)} imports stillhere.{target}")
    assert not violations, "\n".join(violations)


def test_contracts_use_only_the_standard_library():
    import sys
    for path in (PACKAGE / "contracts").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in sys.stdlib_module_names or root == "__future__", f"{path.name} imports {root}"


def _module_import_graph():
    """Edges between stillhere submodules (file level), resolving relative imports."""
    graph = {}
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE.parent).with_suffix("")
        parts = list(rel.parts)
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        package = ".".join(parts[:-1])
        edges = graph.setdefault(name, set())
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")[: len(package.split(".")) - node.level + 1]
                    prefix = ".".join(base + ([node.module] if node.module else []))
                    targets = [prefix] + [f"{prefix}.{a.name}" for a in node.names]
                else:
                    targets = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            edges.update(t for t in targets if t and t.startswith("stillhere"))
    return {k: {t for t in v if t in graph and t != k} for k, v in graph.items()}


def _find_cycle(graph):
    state, stack = {}, []

    def visit(node):
        state[node] = "active"
        stack.append(node)
        for nxt in sorted(graph[node]):
            if state.get(nxt) == "active":
                return stack[stack.index(nxt):] + [nxt]
            if nxt not in state and (cycle := visit(nxt)):
                return cycle
        state[node] = "done"
        stack.pop()
        return None

    for node in sorted(graph):
        if node not in state and (cycle := visit(node)):
            return cycle
    return None


def test_allowed_dependency_graph_is_acyclic():
    assert _find_cycle({k: set(v) for k, v in ALLOWED.items()}) is None


def test_actual_imports_have_no_cycles():
    graph = _module_import_graph()
    assert "stillhere.contracts.inference" in graph and "stillhere.contracts.memory" in graph["stillhere.contracts.inference"]
    cycle = _find_cycle(graph)
    assert cycle is None, " -> ".join(cycle)


def test_cycle_detector_detects_cycles():
    assert _find_cycle({"a": {"b"}, "b": {"a"}}) == ["a", "b", "a"]
