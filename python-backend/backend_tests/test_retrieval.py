from backend.retrieval import BM25Retriever


def test_matching_document_is_ranked_first():
    chunks = [
        {"id": "1", "text": "分页式 KV Cache 可以减少显存碎片并提高复用效率。"},
        {"id": "2", "text": "关系数据库使用索引提高查询速度。"},
    ]
    result = BM25Retriever().search("KV Cache 显存优化", chunks)
    assert result[0]["id"] == "1"


def test_no_match_returns_empty_list():
    assert BM25Retriever().search("天气", [{"id": "1", "text": "模型推理"}]) == []
