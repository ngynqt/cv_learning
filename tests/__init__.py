tests/__init__.py:

 

tests/test_hello_world.py:
def test_hello_world():
    assert "hello" == "hello"

pytest.ini:
[pytest]
testpaths = tests