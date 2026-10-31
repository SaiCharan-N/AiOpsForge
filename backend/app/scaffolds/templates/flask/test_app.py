from app import app


def test_home_page_renders():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
