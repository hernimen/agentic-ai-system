def test_chat_completions_smoke(client):
    response = client.post(
        "/api/v1/chat/completions",
        json={"message": "hello", "thread_id": "t1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"response": "stub response", "thread_id": "t1"}
