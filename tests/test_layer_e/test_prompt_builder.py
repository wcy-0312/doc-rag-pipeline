import pytest
from layer_e.models import EvidenceItem
from layer_e.prompt_builder import build, build_multimodal_messages


@pytest.fixture
def single_evidence():
    return [
        EvidenceItem(
            id="ev-001",
            chunk_id="chunk-001",
            content="建議每 8 小時換藥一次，並觀察傷口狀況。",
            retrieval_weight=0.98,
            source_pages=[3],
            source_tool="azure_di",
        )
    ]


@pytest.fixture
def multi_evidence():
    return [
        EvidenceItem(
            id="ev-001",
            chunk_id="chunk-001",
            content="建議每 8 小時換藥一次。",
            retrieval_weight=0.98,
            source_pages=[3, 5],
            source_tool="azure_di",
        ),
        EvidenceItem(
            id="ev-002",
            chunk_id="chunk-002",
            content="消毒傷口前須確認過敏史。",
            retrieval_weight=0.85,
            source_pages=[],
            source_tool="docling",
        ),
    ]


def test_build_returns_dict_with_system_and_user_keys(single_evidence):
    result = build(single_evidence, "如何換藥？")
    assert isinstance(result, dict)
    assert "system" in result
    assert "user" in result


def test_system_contains_evidence_heading(single_evidence):
    result = build(single_evidence, "如何換藥？")
    assert "[E1]" in result["system"]


def test_user_equals_query(single_evidence):
    query = "如何換藥？"
    result = build(single_evidence, query)
    assert result["user"] == query


def test_evidence_block_includes_pages(single_evidence):
    result = build(single_evidence, "test")
    assert "第 3 頁" in result["system"]
    assert "建議每 8 小時換藥一次" in result["system"]


def test_empty_source_pages_omits_page_reference():
    evidence = [
        EvidenceItem(
            id="ev-001",
            chunk_id="chunk-001",
            content="無頁碼資訊的內容。",
            retrieval_weight=0.75,
            source_pages=[],
            source_tool="docling",
        )
    ]
    result = build(evidence, "test")
    assert "來源頁" not in result["system"]
    assert "無頁碼資訊的內容" in result["system"]


def test_multiple_evidence_items_all_appear(multi_evidence):
    result = build(multi_evidence, "test")
    assert "[E1]" in result["system"]
    assert "[E2]" in result["system"]


def test_evidence_content_appears_in_system(single_evidence):
    result = build(single_evidence, "test")
    assert "建議每 8 小時換藥一次，並觀察傷口狀況。" in result["system"]


def test_build_multimodal_messages_no_images():
    msgs = build_multimodal_messages("sys", "query", [])
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "query"}


def test_build_multimodal_messages_with_image(tmp_path):
    img = tmp_path / "p1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    msgs = build_multimodal_messages("sys", "query", [str(img)])
    assert len(msgs) == 2
    user_content = msgs[1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "query"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_multimodal_messages_caps_at_four_images(tmp_path):
    imgs = []
    for i in range(6):
        p = tmp_path / f"p{i}.png"
        p.write_bytes(b"\x89PNG")
        imgs.append(str(p))
    msgs = build_multimodal_messages("sys", "query", imgs)
    user_content = msgs[1]["content"]
    image_blocks = [b for b in user_content if b.get("type") == "image_url"]
    assert len(image_blocks) == 4


def test_build_multimodal_messages_jpeg_mime(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    msgs = build_multimodal_messages("sys", "query", [str(img)])
    url = msgs[1]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
