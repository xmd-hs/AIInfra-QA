from backend.text import chunk_text, tokenize


def test_chinese_tokenization_contains_bigram():
    assert "推理" in tokenize("大模型推理调度")


def test_long_text_is_chunked():
    chunks = chunk_text("这是用于测试的知识内容。" * 100, size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) >= 10 for chunk in chunks)
