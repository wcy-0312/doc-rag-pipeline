from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    node_id: str
    title: str
    start_page: int | None
    end_page: int | None
    summary: str       # LLM 生成；非葉節點才有意義，葉節點為 ""
    content: str       # 聚合段落文字；只有葉節點才有，非葉節點為 ""
    children: list[TreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def page_range(self) -> tuple[int | None, int | None]:
        return (self.start_page, self.end_page)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "content": self.content,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        return cls(
            node_id=d["node_id"],
            title=d["title"],
            start_page=d.get("start_page"),
            end_page=d.get("end_page"),
            summary=d.get("summary", ""),
            content=d.get("content", ""),
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )


@dataclass
class TreeSearchResult:
    query: str
    matched_nodes: list[TreeNode]
    traversal_path: list[list[str]]   # 每個 matched node 對應一條路徑，如 ["根", "治療", "III期"]


@dataclass
class CrossTreeResult:
    query: str
    guideline_nodes: list[TreeNode]
    patient_nodes: list[TreeNode]
    synthesis: str                    # LLM 跨樹比對結論
